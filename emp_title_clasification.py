"""
Normalización y clasificación de `emp_title` por ámbito profesional.

Salidas:
- emp_tipo:   'puesto' | 'empresa' | 'desconocido'
- emp_ambito: sector/ámbito (o 'puesto_sin_sector', 'empresa_sin_sector', 'otros', 'desconocido')

Las reglas se construyeron mirando SOLO la tabla de frecuencias de train.
Son reglas fijas (no aprenden de los datos), por lo que se aplican igual a train, val y test.
"""
import re
import numpy as np
import pandas as pd

VALORES_DESCONOCIDO = {"", "unknown", "desconocido", "na", "n a", "none", "test"}

# ---------------------------------------------------------------------------
# 1) NORMALIZACIÓN
# ---------------------------------------------------------------------------
# Sustituciones aplicadas sobre el texto ya en minúsculas y sin puntuación.
# Incluye abreviaturas y erratas frecuentes observadas en train.
SUSTITUCIONES = {
    # Expresiones que deben resolverse antes que las abreviaturas sueltas
    r"\bsys admin\b": "systems administrator", r"\bsysadmin\b": "systems administrator",
    r"\bsystems? admin\b": "systems administrator", r"\bnetwork admin\b": "network administrator",
    r"\btech support\b": "technical support", r"\bvet tech\b": "veterinary technician",
    # Abreviaturas
    r"\bsr\b": "senior", r"\bjr\b": "junior",
    r"\bmgr\b": "manager", r"\bmngr\b": "manager", r"\bmgmt\b": "management", r"\bmgt\b": "management",
    r"\basst\b": "assistant", r"\bassit\b": "assistant", r"\badmin\b": "administrative",
    r"\bexec\b": "executive", r"\bdept\b": "department", r"\bdir\b": "director",
    r"\bsupv\b": "supervisor", r"\bsupvr\b": "supervisor", r"\bsupt\b": "superintendent",
    r"\brep\b": "representative", r"\breps\b": "representative",
    r"\btech\b": "technician", r"\btechs\b": "technician",
    r"\bengr\b": "engineer", r"\beng\b": "engineer",
    r"\bacct\b": "accountant", r"\bacctg\b": "accounting",
    r"\bvp\b": "vice president", r"\bavp\b": "assistant vice president", r"\bsvp\b": "senior vice president",
    r"\bevp\b": "executive vice president", r"\bpres\b": "president",
    r"\bgm\b": "general manager", r"\bagm\b": "assistant general manager",
    r"\bhr\b": "human resources", r"\bcsr\b": "customer service representative",
    r"\bcdl\b": "truck driver", r"\botr\b": "truck driver",
    r"(?<!toys r )\bus\b": "united states", r"\busa\b": "united states",
    r"\buniv\b": "university",
    # Erratas frecuentes
    r"\bmanger\b": "manager", r"\bmanagaer\b": "manager", r"\bmanageer\b": "manager", r"\bmanagr\b": "manager",
    r"\bmanagment\b": "management",
    r"\bsupervisior\b": "supervisor", r"\bsupervisorr\b": "supervisor", r"\bsupervsior\b": "supervisor",
    r"\bforman\b": "foreman", r"\bformen\b": "foreman",
    r"\bmecanic\b": "mechanic", r"\bmachanic\b": "mechanic", r"\bmechanics\b": "mechanic",
    r"\bmaintanance\b": "maintenance", r"\bmaintainance\b": "maintenance", r"\bmaintenence\b": "maintenance",
    r"\bmaintance\b": "maintenance", r"\bmantenance\b": "maintenance",
    r"\btracher\b": "teacher", r"\btecher\b": "teacher", r"\bteachers\b": "teacher",
    r"\bnures\b": "nurse", r"\bnurses\b": "nurse",
    r"\bdriver s\b": "driver", r"\bdrivers\b": "driver",
    r"\bowners\b": "owner", r"\bco owner\b": "owner", r"\bcoowner\b": "owner",
    r"\bco founder\b": "founder", r"\bcofounder\b": "founder",
    r"\btechnition\b": "technician", r"\btechnitian\b": "technician", r"\btechnican\b": "technician",
    r"\btechician\b": "technician", r"\btechnicain\b": "technician",
    r"\bmachinest\b": "machinist", r"\boperater\b": "operator", r"\belectrican\b": "electrician",
    r"\bsuperintendant\b": "superintendent", r"\bmananger\b": "manager", r"\bmaneger\b": "manager",
    r"\badminstrator\b": "administrator", r"\blaboror\b": "laborer",
    r"\bbook keeper\b": "bookkeeper", r"\bbookeeper\b": "bookkeeper", r"\bhouse keeping\b": "housekeeping",
    r"\bhouse keeper\b": "housekeeper", r"\btruckdriver\b": "truck driver", r"\btrucker\b": "truck driver",
    r"\bmailhandler\b": "mail handler", r"\bmailman\b": "mail carrier", r"\bforemen\b": "foreman",
    r"\bhuman resource\b": "human resources", r"\bteamleader\b": "team leader", r"\bcare taker\b": "caretaker",
    r"\bhandy man\b": "handyman", r"\bsale\b": "sales", r"\baccount payable\b": "accounts payable",
    r"\baccount receivable\b": "accounts receivable", r"\bmaint\b": "maintenance", r"\bmaintence\b": "maintenance",
    r"\bmaintainence\b": "maintenance", r"\bmachine operater\b": "machine operator",
    r"\bhairdresser\b": "hair stylist", r"\bhairstylist\b": "hair stylist", r"\bsgt\b": "sergeant",
    r"\bssgt\b": "staff sergeant", r"\bssg\b": "staff sergeant", r"\bsfc\b": "sergeant first class",
    r"\borderfiller\b": "order filler", r"\btool maker\b": "toolmaker", r"\bexpediter\b": "expeditor",
    r"\bwal mart\b": "walmart", r"\bmamager\b": "manager", r"\bmanagers\b": "manager",
    r"\benginner\b": "engineer", r"\bengineeer\b": "engineer", r"\bmaintenace\b": "maintenance",
    r"\bwharehouse\b": "warehouse", r"\bwarehous\b": "warehouse", r"\bprinciple\b": "principal",
    r"\baccounts receivables\b": "accounts receivable", r"\bmeatcutter\b": "meat cutter",
    r"\bbusdriver\b": "bus driver", r"\bgrounds keeper\b": "groundskeeper", r"\bmachinists\b": "machinist",
    r"\bsupervisors\b": "supervisor", r"\banalysts\b": "analyst", r"\blongshoremen\b": "longshoreman",
    r"\btoys r us\b": "toys r us", r"\bteksystems\b": "tek systems",
    r"\bsargent\b": "sergeant", r"\bsargeant\b": "sergeant", r"\bparnter\b": "partner",
    r"\bonwer\b": "owner", r"\bower\b": "owner", r"\b0wner\b": "owner", r"\bownder\b": "owner",
    r"\bcheff\b": "chef", r"\bcasher\b": "cashier", r"\bcarpinter\b": "carpenter",
    r"\bcarpintero\b": "carpenter", r"\boperador\b": "operator", r"\bmantenimiento\b": "maintenance",
    r"\bmecanico\b": "mechanic", r"\bcosmotologist\b": "cosmetologist", r"\blettercarrier\b": "letter carrier",
    r"\bmailcarrier\b": "mail carrier", r"\bsalesmanager\b": "sales manager", r"\bvicepresident\b": "vice president",
    r"\bcrewleader\b": "crew leader", r"\bleadperson\b": "lead", r"\bcasemanager\b": "case manager",
    r"\bjourneymen\b": "journeyman", r"\bjourney man\b": "journeyman", r"\blinemen\b": "lineman",
    r"\bsalesmen\b": "salesman", r"\bsaleman\b": "salesman", r"\bwarehousemen\b": "warehouseman",
    r"\bderrickhand\b": "derrick hand", r"\bdeck hand\b": "deckhand", r"\bbody man\b": "bodyman",
    r"\bmachine op\b": "machine operator", r"\bmachine oper\b": "machine operator", r"\baccts\b": "accounts",
    r"\bpayables\b": "payable", r"\breceivables\b": "receivable", r"^a p$": "accounts payable",
    r"^a r$": "accounts receivable", r"^ap$": "accounts payable", r"^ar$": "accounts receivable",
    r"\bhome aid\b": "home health aide", r"\bhistotech\b": "histotechnician", r"\bmech\b": "mechanic",
    r"\bpolicemen\b": "policeman", r"\btoolpusher\b": "tool pusher", r"\bfloor hand\b": "floorhand",
    r"\bcounterperson\b": "counterman", r"\bmaintanence\b": "maintenance", r"\bmaintaince\b": "maintenance",
    r"\bcaptian\b": "captain", r"\bmanaget\b": "manager", r"\breciever\b": "receiver", r"\brecieving\b": "receiving", r"\bcostodian\b": "custodian", r"\bteachet\b": "teacher", r"\bsam s club\b": "sams club", r"\bmacy s\b": "macys",
    r"\blowe s\b": "lowes", r"\bmcdonald s\b": "mcdonalds", r"\bkohl s\b": "kohls",
    r"\bat and t\b": "att", r"\bt mobile\b": "tmobile",
}
_SUSTITUCIONES_COMPILADAS = [(re.compile(p), s) for p, s in SUSTITUCIONES.items()]


def _vocabulario_reglas():
    """Palabras (>= 5 letras) que aparecen en las reglas: es el vocabulario 'correcto'."""
    texto = " ".join(p for r in REGLAS_AMBITO for p in r[1:]) + " " + PATRON_PUESTO_GENERICO + " " + PATRON_ROL
    return {w for w in re.findall(r"[a-z]{5,}", texto)}


# Palabras que no deben "corregirse" aunque se parezcan a una del vocabulario
PALABRAS_PROTEGIDAS = {
    "associates", "enterprises", "partners", "holdings", "industries", "international", "america",
    "systems", "services", "solutions", "technologies", "companies", "products", "networks",
    # Palabras reales de train que se parecen a una palabra del vocabulario pero significan otra cosa
    "superior", "engine", "count", "nation", "heath", "directory", "directv", "independence", "powder",
    "humane", "curative", "baking", "shopping", "marking", "planet", "newyork", "maternal", "maine",
    "melter", "moran", "panther", "salmon", "saloon", "landing", "lansing", "landrys", "stater", "staten",
    "factor", "factors", "integral", "interval", "projection", "prosecution", "transformation",
    "interventional", "applicator", "supper", "drier", "close", "patent", "country", "mobile", "floral",
    "resident", "retention", "gardner", "commonhealth", "trailer", "foster", "water", "internet",
    "counter", "finishing", "electronic", "product", "process", "design", "person", "custom", "motor",
}


def _proponer_correcciones(tokens, vocab, cutoff):
    """Para cada token propone la palabra del vocabulario más parecida (misma inicial)."""
    import difflib
    por_inicial = {}
    for w in vocab:
        por_inicial.setdefault(w[0], []).append(w)
    correccion = {}
    for t in tokens:
        m = difflib.get_close_matches(t, por_inicial.get(t[0], []), n=1, cutoff=cutoff)
        if m:
            correccion[t] = m[0]
    return correccion


def normalizar_emp_title(s: pd.Series) -> pd.Series:
    """Minúsculas, '&' -> ' and ', abreviaturas con puntos unidas (p.c. -> pc),
    resto de signos -> espacio, espacios colapsados, abreviaturas y erratas corregidas."""
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", regex=True)  # p.c. -> pc. ; u.s. -> us.
    s = s.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    for patron, sustituto in _SUSTITUCIONES_COMPILADAS:
        s = s.str.replace(patron, sustituto, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
# 2) REGLAS DE ÁMBITO (el orden importa: gana la primera coincidencia)
#    Cada patrón cubre tanto puestos ("registered nurse") como nombres de
#    organizaciones que revelan el sector ("memorial hospital", "wells fargo").
# ---------------------------------------------------------------------------
def _r(*partes):
    return r"\b(?:" + "|".join(partes) + r")\b"

REGLAS_AMBITO = [
    # --- Excepciones específicas que deben resolverse antes que las reglas generales ---
    ("transporte_logistica", _r(r"air traffic", r"locomotive", r"railroad", r"railway", r"rail road",
                                r"school bus", r"bus driver", r"bus operator")),
    ("ventas_comercio", _r(r"trader joe\w*", r"dunkin donuts")),
    ("finanzas_seguros", _r(r"loan officer", r"credit officer", r"bank officer", r"trust officer",
                            r"financial officer", r"chief financial", r"cfo", r"compliance officer")),
    ("educacion_investigacion", _r(r"school counselor", r"guidance counselor", r"school psychologist",
                                   r"assistant principal", r"vice principal", r"school principal",
                                   r"elementary principal", r"high school principal"),
                                _r(r"engineer", r"consultant", r"scientist", r"analyst", r"architect",
                                   r"investigator", r"partner", r"software", r"data", r"sales", r"program")),
    ("servicios_personales_limpieza", _r(r"massage", r"massage therapist", r"environmental services", r"evs", r"nail technician", r"nail tech")),
    ("hosteleria_ocio", _r(r"table games", r"casino", r"gaming", r"dealer"),
                        _r(r"car", r"auto", r"automotive", r"parts", r"equipment", r"dealership", r"bank", r"services", r"wachovia")),
    ("tecnologia_telecom", _r(r"software", r"developer", r"programmer", r"it", r"information technology",
                              r"information systems", r"information security", r"cyber", r"network",
                              r"system administrator", r"systems administrator", r"systems engineer",
                              r"system engineer", r"systems analyst", r"system analyst", r"database", r"dba",
                              r"web", r"computer", r"help desk", r"helpdesk", r"desktop support",
                              r"technical support", r"tech support", r"devops", r"cloud", r"sap", r"java",
                              r"net developer", r"qa analyst", r"qa engineer", r"qa tester", r"software tester",
                              r"data analyst", r"data scientist", r"data engineer", r"data architect",
                              r"solutions architect", r"software architect", r"enterprise architect",
                              r"application developer", r"applications developer", r"telecom", r"telecommunications",
                              r"microsoft", r"apple", r"intel", r"google", r"ibm", r"oracle", r"cisco", r"dell",
                              r"hewlett packard", r"hp", r"accenture", r"verizon", r"att", r"comcast", r"sprint",
                              r"tmobile", r"computer sciences", r"technology", r"technologies", r"webmaster", r"booz allen", r"saic", r"time warner", r"xerox", r"cable splicer", r"cable", r"scrum master", r"genius", r"centurylink", r"emc", r"ncr", r"citrix", r"avaya", r"infosys", r"infor", r"caci", r"sra international", r"first data", r"fiserv", r"tek systems", r"jack henry", r"cerner",
                              r"fujitsu", r"cablevision", r"directv", r"windstream", r"alcatel", r"ericsson",
                              r"netapp", r"compucom", r"ciso", r"systems integrator", r"pitney bowes", r"qualcomm",
                              r"vmware")),
    ("sanidad", _r(r"medical biller", r"medical billing", r"medical coder", r"medical coding",
                   r"nurse", r"nursing", r"rn", r"lpn", r"lvn", r"cna", r"crna", r"arnp", r"aprn", r"fnp", r"np",
                   r"physician", r"doctor", r"md", r"surgeon", r"surgical", r"surgery", r"medical", r"medicine",
                   r"pharmacy", r"pharmacist", r"pharmaceutical", r"therapist", r"therapy", r"dental", r"dentist",
                   r"hygienist", r"orthodont\w*", r"clinical", r"clinic", r"hospital", r"health", r"healthcare",
                   r"radiolog\w*", r"radiology", r"x ray", r"xray", r"sonograph\w*", r"ultrasound", r"mri", r"ct",
                   r"phlebotom\w*", r"paramedic", r"emt", r"caregiver", r"care giver", r"patient", r"hospice",
                   r"dialysis", r"hemodialysis", r"anesthe\w*", r"cardio\w*", r"chiropract\w*", r"optometr\w*",
                   r"optician", r"veterinar\w*", r"vet tech", r"psychologist", r"psychiatr\w*", r"behavioral",
                   r"mental health", r"rehab", r"rehabilitation", r"respiratory", r"pediatric\w*", r"ob",
                   r"physical therapist", r"occupational therapist", r"pct", r"home health", r"medical center",
                   r"kaiser", r"kaiser permanente", r"mayo clinic", r"physician assistant", r"pa c", r"speech", r"pathologist", r"language pathologist", r"psychotherapist", r"radiographer", r"dietitian", r"dietician", r"dietary", r"nutritionist", r"audiologist", r"transcriptionist", r"podiatrist", r"mammograph\w*", r"cat scan", r"sleep technologist", r"epidemiologist", r"care provider", r"home care", r"home attendant", r"direct care", r"medical assistant", r"nurse aide", r"cancer", r"oncology", r"ambulance", r"hygenist", r"behavior interventionist", r"personal care", r"urgent care", r"histologist", r"gastro\w*", r"histotechnician", r"cytotechnologist",
                   r"perfusionist", r"echocardiograph\w*", r"acupuncturist", r"exercise physiologist",
                   r"neuropsychologist", r"rpsgt", r"dosimetrist", r"vascular", r"crnp", r"ophthalm\w*",
                   r"neurophysiolog\w*", r"infection preventionist", r"biomed", r"lpta", r"efda", r"bcba",
                   r"admitting", r"ems", r"davita", r"upmc", r"johnson and johnson", r"express scripts",
                   r"covidien", r"bristol myers", r"abbott", r"thermo fisher", r"blue cross", r"unitedhealthcare",
                   r"hca", r"cota", r"rvt", r"mortician", r"hospitalist", r"histotechnologist", r"ultrasonographer", r"compounder", r"certified coder", r"medtronic", r"beckman coulter", r"pfizer", r"merck", r"genentech", r"unitedhealth", r"united health", r"wellpoint", r"mckesson", r"quest diagnostics", r"labcorp")),
    ("educacion_investigacion", _r(r"teacher", r"teaching", r"professor", r"instructor", r"educator",
                                   r"education", r"educational", r"school", r"schools", r"university",
                                   r"college", r"academy", r"academic", r"faculty", r"lecturer", r"tutor",
                                   r"paraprofessional", r"paraeducator", r"para", r"substitute", r"preschool",
                                   r"pre k", r"daycare", r"child care", r"childcare", r"dean", r"librarian",
                                   r"library", r"coach", r"isd", r"boces", r"registrar", r"admissions",
                                   r"postdoc", r"post doc", r"postdoctoral", r"research", r"researcher",
                                   r"superintendent of schools", r"montessori", r"department chair", r"graduate student", r"student", r"ucla", r"institute", r"apollo group", r"scholastic", r"ucsf", r"adjunct", r"bursar", r"nyc doe", r"archivist", r"curator", r"bus monitor")),
    ("seguridad_defensa", _r(r"police", r"sheriff", r"deputy sheriff", r"correctional", r"corrections",
                             r"correction officer", r"detention", r"jail", r"jailer", r"prison", r"probation",
                             r"parole", r"trooper", r"detective", r"firefighter", r"fire fighter", r"fireman",
                             r"military", r"army", r"navy", r"air force", r"usaf", r"usmc", r"marine corps",
                             r"united states marine", r"marines", r"national guard", r"coast guard", r"soldier",
                             r"sergeant", r"lieutenant", r"colonel", r"nco", r"ncoic", r"infantry",
                             r"e ?[1-9]", r"staff sergeant", r"corporal", r"enlisted", r"active duty",
                             r"patrolman", r"border patrol", r"special agent", r"fbi",
                             r"federal bureau of investigation", r"homeland security", r"secret service",
                             r"tsa", r"screener", r"law enforcement", r"department of defense", r"dod",
                             r"nypd", r"brinks", r"g4s", r"tsgt", r"infantryman", r"policeman", r"airman", r"corpsman", r"1sg", r"lt col", r"cw[1-5]",
                             r"po[1-3]", r"lcpo", r"msg", r"cpt", r"yeoman", r"allied barton", r"bailiff")),
    ("seguridad_defensa", _r(r"police", r"sheriff", r"deputy sheriff", r"correctional", r"corrections",
                             r"correction officer", r"detention", r"jail", r"prison", r"probation", r"parole",
                             r"trooper", r"detective", r"investigator", r"firefighter", r"fire", r"fire fighter",
                             r"military", r"army", r"navy", r"air force", r"usaf", r"usmc", r"marine corps",
                             r"marines", r"national guard", r"soldier", r"sergeant", r"lieutenant", r"colonel",
                             r"nco", r"security", r"border patrol", r"customs", r"special agent", r"fbi",
                             r"federal bureau of investigation", r"homeland security", r"tsa", r"public safety",
                             r"law enforcement", r"department of defense", r"dod", r"deputy", r"officer", r"e ?[1-9]", r"staff sergeant", r"corporal", r"enlisted", r"active duty", r"patrolman", r"patrol", r"armed guard", r"guard", r"loss prevention", r"asset protection", r"fireman", r"nypd", r"park ranger", r"ranger", r"telecommunicator", r"911", r"commander"),
                          _r(r"executive", r"operating", r"information", r"technology", r"marketing",
                             r"financial", r"medical", r"nursing", r"human resources", r"compliance",
                             r"admissions", r"program", r"project", r"loan", r"credit", r"claims",
                             r"benefits", r"procurement", r"contracting", r"contract", r"director",
                             r"clerk", r"treasurer", r"assessor", r"commissioner", r"administrator",
                             r"privacy", r"development", r"business", r"bank", r"trust", r"investment",
                             r"revenue", r"accounting", r"academic", r"research", r"training", r"personnel",
                             r"administrative", r"purchasing", r"lending", r"relationship", r"branch", r"quality", r"certification", r"operations")),
    ("legal", _r(r"attorney", r"lawyer", r"paralegal", r"legal", r"law", r"counsel", r"judge", r"judicial",
                 r"court", r"litigation", r"law firm", r"esquire", r"esq", r"notary", r"prosecutor", r"magistrate", r"courts")),
    ("finanzas_seguros", _r(r"accountant", r"accounting", r"accounts payable", r"accounts receivable", r"cpa",
                            r"bookkeeper", r"bookkeeping", r"controller", r"comptroller", r"auditor", r"audit",
                            r"finance", r"financial", r"bank", r"banker", r"banking", r"teller", r"loan", r"loans",
                            r"lending", r"mortgage", r"underwriter", r"underwriting", r"credit", r"credit union",
                            r"investment", r"investments", r"broker", r"brokerage", r"insurance", r"claims",
                            r"adjuster", r"actuar\w*", r"tax", r"payroll", r"billing", r"collections",
                            r"collector", r"escrow", r"treasury", r"treasurer", r"wealth", r"portfolio", r"trader",
                            r"trading", r"equity", r"fund", r"capital", r"budget", r"cashier"
                            r"(?= check)", r"wells fargo", r"chase", r"jpmorgan", r"jp morgan", r"citi",
                            r"citibank", r"citigroup", r"pnc", r"td bank", r"capital one", r"american express",
                            r"amex", r"suntrust", r"bb and t", r"fifth third", r"comerica", r"morgan stanley",
                            r"goldman sachs", r"charles schwab", r"schwab", r"fidelity", r"merrill lynch",
                            r"edward jones", r"ernst and young", r"kpmg", r"pricewaterhousecoopers", r"pwc",
                            r"deloitte", r"state farm", r"allstate", r"geico", r"usaa", r"prudential", r"metlife",
                            r"aflac", r"progressive", r"nationwide", r"liberty mutual", r"travelers", r"aetna",
                            r"cigna", r"humana", r"farmers insurance", r"paychex", r"adp", r"lender", r"commercial lender", r"closer", r"economist", r"bny mellon", r"mellon", r"the hartford", r"hartford", r"funder", r"tiaa", r"ubs", r"visa", r"vanguard", r"grant thornton", r"bdo", r"bloomberg", r"northwestern mutual", r"aig",
                            r"northern trust", r"hsbc", r"barclays", r"blackrock", r"fannie mae", r"new york life",
                            r"bbva", r"stockbroker", r"paraplanner", r"loss mitigation", r"payment poster",
                            r"collection", r"santander")),
    ("inmobiliaria", _r(r"realtor", r"real estate", r"property manager", r"property management",
                        r"property", r"properties", r"leasing", r"apartment", r"apartments", r"realty",
                        r"title", r"appraiser", r"abstractor", r"jones lang lasalle", r"landlord", r"cbre")),
    ("atencion_cliente", _r(r"customer services?", r"customer care", r"customer experience", r"convergys", r"customer support", r"customer representative",
                            r"customer rep", r"call center", r"client services", r"member services",
                            r"guest services", r"customer success", r"customer relations", r"client relations", r"member service")),
    ("transporte_logistica", _r(r"driver", r"truck", r"trucking", r"delivery", r"courier", r"dispatcher",
                                r"dispatch", r"pilot", r"flight attendant", r"flight", r"airline", r"airlines",
                                r"aviation", r"airport", r"conductor", r"bus operator", r"transit", r"logistics",
                                r"warehouse", r"warehouseman", r"forklift", r"fork lift", r"material handler",
                                r"materials handler", r"shipping", r"receiving", r"shipper", r"order selector",
                                r"order picker", r"picker", r"freight", r"carrier", r"mail", r"postal",
                                r"postmaster", r"letter carrier", r"usps", r"united states postal service",
                                r"post office", r"ups", r"united parcel", r"fedex", r"yrc", r"transportation",
                                r"fleet", r"chauffeur", r"longshoreman", r"port", r"amtrak", r"union pacific",
                                r"bnsf", r"csx", r"southwest airlines", r"delta", r"united airlines",
                                r"american airlines", r"moving", r"mail handler", r"package handler", r"loader", r"unloader", r"sorter", r"order filler", r"dockworker", r"dock", r"deckhand", r"tankerman", r"yardmaster", r"signalman", r"carman", r"signal maintainer", r"transporter", r"logistician", r"supply chain", r"distribution", r"expeditor", r"ramp agent", r"ramp", r"inventory", r"stock", r"packing", r"parking", r"messenger", r"teamster", r"hauler", r"chauffer", r"yard jockey", r"switchman", r"motorman", r"trainman", r"trainmaster",
                                r"trackworker", r"trackman", r"loadmaster", r"able seaman", r"able bodied seaman",
                                r"skycap", r"fueler", r"preloader", r"order puller", r"puller", r"mover",
                                r"navigator", r"estes", r"ground ops", r"logistic", r"material handling",
                                r"stocking", r"transport", r"norfolk southern", r"derrick" r"(?= operator)", r"longshoreman", r"handler", r"supply", r"airways", r"hertz", r"avis", r"rent a car", r"enterprise holdings", r"distributor")),
    ("hosteleria_ocio", _r(r"server", r"waiter", r"waitress", r"bartender", r"barista", r"cook", r"chef",
                           r"sous chef", r"kitchen", r"restaurant", r"food service", r"foodservice", r"dishwasher",
                           r"host", r"hostess", r"hotel", r"banquet", r"catering", r"caterer", r"concierge",
                           r"front desk", r"resort", r"golf", r"travel", r"tourism", r"starbucks", r"mcdonalds",
                           r"disney", r"entertainment", r"bakery", r"baker", r"butcher", r"deli", r"cafe",
                           r"food", r"beverage", r"bar", r"busser", r"bellman", r"doorman", r"valet", r"steward", r"hospitality", r"cake decorator", r"decorator", r"meat cutter", r"meat", r"butcher", r"brewer", r"slot", r"reservations", r"tour", r"tour guide", r"prep", r"compass group", r"sodexo", r"aramark", r"marriott", r"mgm", r"brinker", r"bakeries", r"dining",
                           r"banquets", r"reservationist", r"barback", r"waitstaff", r"sommelier", r"cooker",
                           r"butler", r"maitre d", r"cafeteria", r"ihop", r"panera", r"pizza hut", r"hilton",
                           r"starwood", r"usher", r"lifeguard", r"rgm")),
    ("ventas_comercio", _r(r"sales", r"salesman", r"saleswoman", r"salesperson", r"account executive",
                           r"retail", r"store", r"stores", r"cashier", r"merchandiser", r"merchandising",
                           r"merchandise", r"buyer", r"purchasing", r"procurement", r"stocker", r"produce",
                           r"grocery", r"business development", r"territory manager", r"showroom",
                           r"walmart", r"target", r"home depot", r"lowes", r"costco", r"sams club", r"kroger",
                           r"safeway", r"publix", r"macys", r"sears", r"best buy", r"kmart", r"walgreens",
                           r"cvs", r"rite aid", r"nordstrom", r"kohls", r"jcpenney", r"j c penney", r"gap",
                           r"amazon", r"car dealer", r"auto dealer", r"dealership", r"parts manager",
                           r"service advisor", r"service writer", r"storekeeper", r"key holder", r"purchaser", r"whole foods", r"jeweler", r"jewelry", r"greeter", r"merchant", r"staples", r"aldi", r"guitar center", r"rent a center", r"toys r us", r"tiffany",
                           r"counterman", r"parts counter", r"personal shopper", r"keyholder", r"florist",
                           r"bed bath", r"albertsons", r"heb", r"carmax", r"neiman marcus", r"fred meyer",
                           r"petsmart", r"big lots", r"stop and shop", r"advance auto", r"sherwin williams",
                           r"grainger", r"barnes and noble", r"bookseller", r"saks", r"bloomingdales",
                           r"supervalu", r"ebay", r"wholesaler", r"internal wholesaler", r"dsm", r"pricing", r"autozone", r"dollar general", r"dollar tree", r"family dollar", r"wholesale", r"reseller", r"market")),
    ("marketing_medios_diseno", _r(r"marketing", r"advertising", r"communications", r"communication",
                                   r"public relations", r"media", r"journalist", r"reporter", r"editor",
                                   r"writer", r"producer", r"designer", r"graphic", r"creative", r"art",
                                   r"artist", r"photographer", r"videographer", r"music", r"musician",
                                   r"broadcast", r"radio", r"television", r"tv", r"film", r"publisher",
                                   r"publishing", r"brand", r"social media", r"content", r"copywriter", r"actor", r"interpreter", r"translator", r"thomson reuters", r"linguist", r"entertainer", r"performer", r"stagehand", r"news", r"anchor", r"public affairs", r"symphony", r"orchestra", r"theatre", r"theater", r"pictures", r"studios", r"zoo", r"museum", r"singer", r"marketer", r"photojournalist", r"dancer", r"dj", r"broadcaster", r"set dresser"),
                                _r(r"engineer", r"engineering", r"electrical", r"mechanical", r"structural",
                                   r"civil", r"cad", r"piping", r"hvac")),
    ("servicios_sociales_religion", _r(r"social worker", r"social work", r"social services", r"caseworker",
                                       r"case worker", r"case manager", r"human services", r"counselor",
                                       r"youth", r"child welfare", r"family services", r"nonprofit",
                                       r"non profit", r"foundation", r"community", r"pastor", r"minister",
                                       r"ministry", r"church", r"chaplain", r"priest", r"rabbi", r"clergy",
                                       r"missionary", r"volunteer", r"advocate", r"direct support", r"mentor", r"organizer", r"rector", r"charities", r"catholic charities", r"fundraiser", r"fundraising", r"eligibility", r"red cross", r"foster parent", r"home visitor", r"ymca", r"ihss", r"qidp")),
    ("administracion_publica", _r(r"city of", r"county of", r"state of", r"town of", r"village of",
                                  r"township", r"county", r"government", r"federal", r"public works",
                                  r"department of", r"municipal", r"commonwealth", r"city", r"irs",
                                  r"social security", r"veterans affairs", r"usda", r"state", r"internal revenue", r"civil service", r"assessor", r"examiner", r"commissioner", r"mayor", r"civil servant", r"diplomat", r"benefit authorizer", r"adjudicator", r"govt", r"gs ?[0-9]+", r"census", r"forest service", r"fish and wildlife", r"house of representatives", r"senate", r"general services administration", r"building official")),
    ("ingenieria_ciencia", _r(r"engineer", r"engineering", r"architect", r"architecture", r"drafter",
                              r"draftsman", r"cad", r"scientist", r"chemist", r"chemistry", r"biologist",
                              r"geologist", r"geoscientist", r"physicist", r"laboratory", r"lab", r"surveyor",
                              r"environmental", r"boeing", r"lockheed", r"lockheed martin", r"raytheon",
                              r"northrop", r"general dynamics", r"microbiologist", r"statistician", r"meteorologist", r"structural designer", r"piping designer", r"cad designer", r"designer", r"hydrogeologist", r"geophysicist", r"hydrologist", r"archaeologist", r"cartographer", r"urs", r"bae systems", r"harris", r"aerospace", r"loral", r"moog")),
    ("construccion_oficios", _r(r"construction", r"contractor", r"carpenter", r"electrician", r"electrical",
                                r"plumber", r"plumbing", r"welder", r"welding", r"hvac", r"painter", r"laborer",
                                r"labor", r"foreman", r"superintendent", r"roofer", r"roofing", r"mason",
                                r"ironworker", r"iron worker", r"millwright", r"pipefitter", r"pipe fitter",
                                r"lineman", r"installer", r"framer", r"drywall", r"concrete", r"heavy equipment",
                                r"equipment operator", r"crane", r"journeyman", r"apprentice", r"landscap\w*",
                                r"groundskeeper", r"estimator", r"inspector", r"builder", r"excavat\w*",
                                r"scaffold\w*", r"glazier", r"sheet metal", r"insulator", r"boilermaker", r"steamfitter", r"sprinkler fitter", r"bricklayer", r"rigger", r"skilled trades", r"cabinet maker", r"tile setter", r"glazer", r"gardener", r"steelworker", r"pipe layer", r"elevator constructor",
                                r"groundman", r"groundsman", r"grounds", r"arborist", r"sawyer", r"woodworker",
                                r"craftsman", r"plaster", r"plasterer", r"taper", r"flagger", r"splicer",
                                r"inside wireman", r"wireman", r"electrian", r"cabinetmaker", r"upholsterer",
                                r"trimmer", r"bechtel", r"aecom")),
    ("construccion_oficios", _r(r"heavy equipment", r"equipment operator", r"crane")),
    ("industria_manufactura", _r(r"machinist", r"machine operator", r"operator", r"production", r"assembler",
                                 r"assembly", r"manufacturing", r"manufacturer", r"plant", r"factory",
                                 r"fabricat\w*", r"press", r"cnc", r"quality", r"quality control",
                                 r"quality assurance", r"packer", r"packaging", r"extruder", r"mill",
                                 r"line worker", r"industries", r"industrial", r"ford", r"chrysler",
                                 r"general motors", r"3m", r"procter and gamble", r"general electric",
                                 r"caterpillar", r"john deere", r"honeywell", r"steel", r"plastics",
                                 r"chemical", r"textile", r"printing", r"printer", r"pressman", r"finisher", r"toolmaker", r"tool and die", r"mixer", r"grinder", r"fitter", r"tester", r"inspection", r"frito lay", r"pepsico", r"coca cola", r"international paper", r"leadman", r"blender", r"packager", r"mold maker", r"polisher", r"feeder", r"die setter", r"siemens", r"dupont", r"eaton", r"abb", r"johnson controls", r"pepsi", r"dr pepper", r"nestle", r"foods",
                                 r"molder", r"die maker", r"cutter", r"bindery", r"prepress", r"laminator", r"presser",
                                 r"caster", r"plater", r"weaver", r"sewer", r"machine tender", r"pattern maker",
                                 r"sample maker", r"palletizer", r"autoworker", r"finishing", r"filler", r"grader",
                                 r"metrologist", r"oiler", r"rr donnelley", r"georgia pacific", r"cargill",
                                 r"unilever", r"millercoors", r"schneider electric", r"texas instruments",
                                 r"pratt and whitney", r"ge", r"winemaker")),
    ("mantenimiento_reparacion", _r(r"mechanic", r"maintenance", r"technician", r"repair", r"service technician",
                                    r"field service", r"facilities", r"facility", r"handyman", r"auto body",
                                    r"collision", r"tire", r"locksmith", r"maintainer", r"handyman", r"repairman", r"detailer", r"bodyman", r"serviceman", r"mechanical")),
    ("servicios_personales_limpieza", _r(r"custodian", r"custodial", r"janitor", r"janitorial", r"housekeeping",
                                         r"housekeeper", r"porter", r"cleaning", r"cleaner", r"stylist", r"hair",
                                         r"barber", r"cosmetolog\w*", r"nail", r"manicurist", r"esthetician",
                                         r"spa", r"salon", r"nanny", r"personal trainer", r"fitness", r"groomer",
                                         r"pet", r"laundry", r"room attendant", r"caretaker", r"sanitation", r"seamstress", r"tailor", r"houseman", r"houseperson", r"aesthetician", r"republic services", r"waste", r"cintas", r"building services", r"mortuary", r"funeral", r"orkin", r"pest control", r"homemaker", r"exterminator", r"babysitter", r"colorist", r"beautician", r"abm")),
    ("sector_primario_energia", _r(r"farm", r"farmer", r"farming", r"ranch", r"rancher", r"agricultur\w*",
                                   r"oil", r"gas", r"oilfield", r"drilling", r"driller", r"rig", r"energy",
                                   r"utility", r"utilities", r"power plant", r"power", r"mining", r"miner",
                                   r"coal", r"petroleum", r"pipeline", r"forestry", r"fishing", r"nursery",
                                   r"coil tubing", r"wireline", r"frac", r"pumper", r"landman", r"forester", r"floorhand", r"meter reader", r"locator", r"chevron", r"edison", r"weatherford", r"roustabout",
                                   r"derrick hand", r"floorman", r"tool pusher", r"blaster", r"schlumberger",
                                   r"halliburton", r"exxonmobil", r"exxon", r"bp", r"phillips 66", r"conocophillips",
                                   r"national grid")),
    ("administracion_oficina_rrhh", _r(r"administrative", r"administrative assistant", r"executive assistant",
                                       r"office", r"office manager", r"secretary", r"receptionist", r"clerical",
                                       r"clerk", r"data entry", r"typist", r"file clerk", r"human resources",
                                       r"recruiter", r"recruiting", r"recruitment", r"talent", r"staffing",
                                       r"personnel", r"benefits", r"office assistant", r"scheduler",
                                       r"scheduling", r"records", r"document control", r"kelly services", r"robert half", r"hrbp", r"employment specialist", r"order entry", r"timekeeper", r"generalist", r"adecco", r"aerotek")),
]

# Puestos genéricos que no revelan sector (se evalúan DESPUÉS de las reglas de ámbito)
PATRON_PUESTO_GENERICO = _r(
    r"manager", r"management", r"owner", r"supervisor", r"director", r"president", r"vice president",
    r"ceo", r"coo", r"cto", r"cio", r"chief", r"executive", r"partner", r"founder", r"consultant", r"analyst",
    r"specialist", r"associate", r"coordinator", r"administrator", r"lead", r"leader", r"team lead",
    r"operations", r"operation", r"project manager", r"program manager", r"general manager", r"principal",
    r"self employed", r"selfemployed", r"business owner", r"entrepreneur", r"representative", r"agent",
    r"assistant", r"officer", r"employee", r"worker", r"staff", r"captain", r"planner", r"trainer",
    r"processor", r"proprietor", r"principal", r"head", r"superviser", r"administration", r"business",
    r"contract", r"senior", r"junior", r"intern", r"trainee", r"helper", r"aide", r"advisor", r"adviser",
    r"liaison", r"inspector", r"investigator", r"auditor", r"estimator", r"buyer", r"programs", r"program",
    r"project", r"projects", r"operator", r"technician", r"specialists", r"retired", r"full time",
    r"part time", r"various", r"multiple", r"none", r"self", r"independent", r"freelance", r"contractor", r"member", r"crew", r"professional", r"boss", r"own", r"manage", r"supervision", r"support", r"technical", r"service", r"expert", r"strategist", r"fellow", r"hourly", r"vendor", r"super", r"facilitator", r"shareholder", r"safety", r"compliance", r"attendant", r"general", r"chairman", r"consulting", r"employed", r"fulltime", r"floater", r"evaluator", r"monitor", r"ambassador", r"negotiator", r"franchisee", r"crew leader", r"dm", r"temp", r"freelancer", r"account", r"training", r"processing"
)

# Señales de que el texto es el nombre de una organización
PATRON_SUFIJO_LEGAL = r"\s(?:inc|incorporated|llc|ltd|limited|corp|corporation|company|co|lp|llp|pc|pllc|plc|pa|na|holdings|enterprises|group|associates|partners|international|intl)$"
PATRON_ORGANIZACION = _r(
    r"university", r"college", r"hospital", r"medical center", r"health system", r"healthcare system",
    r"school district", r"public schools", r"schools", r"isd", r"boces", r"academy", r"bank", r"credit union",
    r"department of", r"city of", r"county of", r"state of", r"town of", r"village of", r"township",
    r"county", r"united states", r"army", r"navy", r"air force", r"marine corps", r"national guard",
    r"foundation", r"church", r"clinic", r"institute", r"(?<!call )(?<!data )(?<!service )center",
    r"technologies", r"industries", r"enterprises", r"agency", r"restaurant",
    r"hotel", r"resort", r"salon", r"realty", r"properties", r"motors", r"airlines",
    r"walmart", r"target", r"home depot", r"lowes", r"costco", r"sams club", r"kroger", r"safeway", r"publix",
    r"macys", r"sears", r"best buy", r"kmart", r"walgreens", r"cvs", r"amazon", r"wells fargo", r"chase",
    r"jpmorgan", r"jp morgan", r"citi", r"citibank", r"capital one", r"american express", r"state farm",
    r"allstate", r"geico", r"usaa", r"kpmg", r"deloitte", r"pwc", r"ernst and young", r"ups", r"fedex",
    r"usps", r"att", r"verizon", r"comcast", r"microsoft", r"apple", r"intel", r"google", r"ibm", r"oracle",
    r"cisco", r"dell", r"boeing", r"lockheed", r"raytheon", r"northrop", r"ford", r"chrysler",
    r"general motors", r"kaiser", r"starbucks", r"mcdonalds", r"disney", r"fbi", r"irs", r"usda", r"amtrak"
)
# Si el texto contiene una de estas palabras, se considera un puesto aunque mencione una organización
PATRON_ROL = _r(
    r"manager", r"management", r"owner", r"supervisor", r"director", r"president", r"teacher", r"nurse",
    r"driver", r"officer", r"clerk", r"assistant", r"specialist", r"technician", r"engineer", r"analyst",
    r"representative", r"agent", r"associate", r"coordinator", r"administrator", r"teller", r"banker",
    r"professor", r"instructor", r"worker", r"operator", r"cashier", r"lead", r"leader", r"consultant",
    r"executive", r"secretary", r"accountant", r"attorney", r"physician", r"therapist", r"mechanic",
    r"recruiter", r"advisor", r"planner", r"inspector", r"investigator", r"processor", r"underwriter",
    r"adjuster", r"auditor", r"counselor", r"chief", r"captain", r"sergeant", r"soldier", r"carrier",
    r"server", r"cook", r"chef", r"stylist", r"designer", r"developer", r"programmer", r"scientist",
    r"pastor", r"partner", r"aide", r"tech", r"electrician", r"plumber", r"welder", r"laborer", r"foreman",
    r"paralegal", r"pharmacist", r"dentist", r"hygienist", r"librarian", r"custodian", r"janitor",
    r"housekeeper", r"bartender", r"realtor", r"broker", r"buyer", r"dispatcher", r"pilot", r"conductor",
    r"machinist", r"assembler", r"estimator", r"superintendent", r"rn", r"lpn", r"cna", r"md", r"ceo",
    r"cfo", r"coo", r"founder", r"head", r"staff", r"employee", r"intern", r"trainer", r"coach", r"deputy",
    r"detective", r"firefighter", r"paramedic", r"emt", r"caregiver", r"tutor", r"dean", r"faculty", r"architect", r"sales", r"producer", r"dealer", r"examiner", r"services", r"service", r"support", r"department", r"student"
)

# Siglas y términos que solo se clasifican cuando son el título completo
EXACTOS = {
    "pa": "sanidad", "ma": "sanidad", "cma": "sanidad", "pca": "sanidad", "hha": "sanidad", "pta": "sanidad",
    "cota": "sanidad", "rda": "sanidad", "rdh": "sanidad", "cst": "sanidad", "mlt": "sanidad", "cpht": "sanidad",
    "stna": "sanidad", "rrt": "sanidad", "psr": "sanidad", "cca": "sanidad", "slp": "sanidad", "coder": "sanidad",
    "biller": "sanidad", "provider": "sanidad", "resident": "sanidad", "clinician": "sanidad",
    "technologist": "sanidad", "medic": "sanidad",
    "csa": "atencion_cliente", "css": "atencion_cliente", "csm": "atencion_cliente", "msr": "atencion_cliente",
    "rsr": "ventas_comercio", "rsa": "ventas_comercio", "asm": "ventas_comercio", "gsm": "ventas_comercio",
    "fsr": "mantenimiento_reparacion", "tso": "seguridad_defensa", "cbpo": "seguridad_defensa",
    "lmt": "servicios_personales_limpieza", "dsp": "servicios_sociales_religion", "qc": "industria_manufactura",
    "qa": "industria_manufactura", "lausd": "educacion_investigacion", "nypd": "seguridad_defensa",
    "major": "seguridad_defensa", "commander": "seguridad_defensa", "enlisted": "seguridad_defensa",
    "active duty": "seguridad_defensa", "corporal": "seguridad_defensa", "guard": "seguridad_defensa",
    "mate": "transporte_logistica", "parts": "ventas_comercio", "stock": "transporte_logistica",
    "inventory": "transporte_logistica", "receiver": "transporte_logistica", "selector": "transporte_logistica",
    "loader": "transporte_logistica", "sorter": "transporte_logistica", "checker": "ventas_comercio",
    "drive": "transporte_logistica", "atcs": "transporte_logistica", "faa": "transporte_logistica",
    "mlo": "finanzas_seguros", "cls": "sanidad", "lna": "sanidad", "ccht": "sanidad", "chha": "sanidad",
    "rma": "sanidad", "lcsw": "servicios_sociales_religion", "ncoic": "seguridad_defensa",
    "lt": "seguridad_defensa", "dhs": "seguridad_defensa", "msgt": "seguridad_defensa",
    "ea": "administracion_oficina_rrhh", "pm": "puesto_sin_sector", "cmo": "marketing_medios_diseno",
    "reception": "administracion_oficina_rrhh", "attendant": "puesto_sin_sector", "design": "marketing_medios_diseno", "principal": "educacion_investigacion",
}

_REGLAS_COMPILADAS = [
    (r[0], re.compile(r[1]), re.compile(r[2]) if len(r) > 2 else None) for r in REGLAS_AMBITO
]
_RE_GENERICO = re.compile(PATRON_PUESTO_GENERICO)
_RE_SUFIJO = re.compile(PATRON_SUFIJO_LEGAL)
_RE_ORG = re.compile(PATRON_ORGANIZACION)
_RE_ROL = re.compile(PATRON_ROL)


def _clasificar_unicos(norm: pd.Series) -> pd.DataFrame:
    """Clasifica un Series de títulos ya normalizados (valores únicos)."""
    desconocido = norm.isin(VALORES_DESCONOCIDO)

    # Tipo de registro: empresa si lleva sufijo legal, o si menciona una organización sin ningún rol
    sufijo = norm.str.contains(_RE_SUFIJO)
    org = norm.str.contains(_RE_ORG)
    rol = norm.str.contains(_RE_ROL)
    es_empresa = sufijo | (org & ~rol)
    tipo = np.where(desconocido, "desconocido", np.where(es_empresa, "empresa", "puesto"))

    # Ámbito: primero coincidencias exactas, después la primera regla que coincida
    ambito = norm.map(EXACTOS).astype("object")
    pendiente = ~desconocido & ambito.isna()
    for amb, incluye, excluye in _REGLAS_COMPILADAS:
        coincide = pendiente & norm.str.contains(incluye)
        if excluye is not None:
            coincide &= ~norm.str.contains(excluye)
        ambito[coincide] = amb
        pendiente &= ~coincide

    generico = pendiente & ~es_empresa & norm.str.contains(_RE_GENERICO)
    ambito[generico] = "puesto_sin_sector"
    pendiente &= ~generico
    ambito[pendiente & es_empresa] = "empresa_sin_sector"
    ambito[pendiente & ~es_empresa] = "otros"
    ambito[desconocido] = "desconocido"

    return pd.DataFrame({"emp_tipo": tipo, "emp_ambito": ambito.values}, index=norm.index)


def _normalizar_por_unicos(emp_title: pd.Series) -> pd.Series:
    """Normaliza solo los valores distintos y después mapea (mucho más rápido)."""
    crudo = emp_title.fillna("").astype(str)
    unicos = pd.Series(crudo.unique())
    mapa = dict(zip(unicos, normalizar_emp_title(unicos)))
    return crudo.map(mapa)


class ClasificadorEmpTitle:
    """Normaliza y clasifica emp_title.

    fit(): aprende SOLO de train las correcciones de erratas (palabras raras en train que se
           parecen a una palabra del vocabulario de las reglas).
    transform(): aplica normalización, correcciones y reglas. Las palabras nuevas que aparezcan
           en val/test (no vistas en train) se corrigen con el mismo criterio de similitud.
    """

    def __init__(self, corregir=True, cutoff=0.88, max_frecuencia=5, ratio_minimo=20, min_longitud=6):
        self.corregir = corregir
        self.cutoff = cutoff
        self.max_frecuencia = max_frecuencia      # palabras con <= N apariciones: siempre candidatas
        self.ratio_minimo = ratio_minimo          # o bien la palabra correcta es >= R veces más frecuente
        self.min_longitud = min_longitud

    def _candidatas(self, tokens):
        t = pd.Series(tokens)
        mask = (t.str.len() >= self.min_longitud) & t.str.isalpha() & ~t.isin(self.vocab_) & ~t.isin(PALABRAS_PROTEGIDAS)
        return t[mask].tolist()

    def fit(self, emp_title: pd.Series):
        self.vocab_ = _vocabulario_reglas()
        self.correcciones_ = {}
        self.tokens_train_ = set()
        if self.corregir:
            conteo = emp_title.fillna("").astype(str).value_counts()
            norm_unicos = normalizar_emp_title(pd.Series(conteo.index))
            tokens = pd.DataFrame({"token": norm_unicos.str.split().values, "n": conteo.values}).explode("token").dropna()
            frecuencia = tokens.groupby("token")["n"].sum()
            self.tokens_train_ = set(frecuencia.index)
            propuestas = _proponer_correcciones(self._candidatas(frecuencia.index), self.vocab_, self.cutoff)
            # Se acepta la corrección si la palabra es rara o si la palabra correcta es mucho más frecuente
            # (una errata frecuente, como 'adminstrative', sigue siendo muy rara frente a 'administrative')
            self.correcciones_ = {
                t: c for t, c in propuestas.items()
                if frecuencia[t] <= self.max_frecuencia
                or frecuencia.get(c, 0) >= self.ratio_minimo * frecuencia[t]
            }
        return self

    def _aplicar_correcciones(self, unicos: pd.Series) -> pd.Series:
        correcciones = dict(self.correcciones_)
        nuevos = set(unicos.str.split().explode().dropna().unique()) - self.tokens_train_
        if nuevos:  # palabras no vistas en train: son raras por definición
            correcciones.update(_proponer_correcciones(self._candidatas(sorted(nuevos)), self.vocab_, self.cutoff))
        return unicos.map(lambda s: " ".join(correcciones.get(w, w) for w in s.split()))

    def transform(self, emp_title: pd.Series) -> pd.DataFrame:
        norm = _normalizar_por_unicos(emp_title)
        unicos = pd.Series(norm.unique())
        corregidos = self._aplicar_correcciones(unicos) if self.corregir else unicos
        tabla = _clasificar_unicos(corregidos)
        tabla.index = unicos.values
        tabla["emp_title_norm"] = corregidos.values
        return pd.DataFrame({
            "emp_title_norm": tabla.loc[norm.values, "emp_title_norm"].values,
            "emp_tipo": tabla.loc[norm.values, "emp_tipo"].values,
            "emp_ambito": tabla.loc[norm.values, "emp_ambito"].values,
        }, index=emp_title.index)
