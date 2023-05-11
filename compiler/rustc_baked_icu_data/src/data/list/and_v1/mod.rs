// @generated
type DataStruct = <::icu_list::provider::AndListV1Marker as ::icu_provider::DataMarker>::Yokeable;
pub fn lookup(locale: &icu_provider::DataLocale) -> Option<&'static DataStruct> {
    static KEYS: [&str; 12usize] =
        ["en", "es", "fr", "it", "ja", "pt", "ru", "tr", "und", "zh", "zh-Hans", "zh-Hant"];
    static DATA: [&DataStruct; 12usize] =
        [&EN, &ES, &FR, &IT, &JA, &PT, &RU, &TR, &UND, &ZH, &ZH, &ZH_HANT];
    KEYS.binary_search_by(|k| locale.strict_cmp(k.as_bytes()).reverse())
        .ok()
        .map(|i| unsafe { *DATA.get_unchecked(i) })
}
static EN: DataStruct = include!("en.rs.data");
static ES: DataStruct = include!("es.rs.data");
static FR: DataStruct = include!("fr.rs.data");
static IT: DataStruct = include!("it.rs.data");
static JA: DataStruct = include!("ja.rs.data");
static PT: DataStruct = include!("pt.rs.data");
static RU: DataStruct = include!("ru.rs.data");
static TR: DataStruct = include!("tr.rs.data");
static UND: DataStruct = include!("und.rs.data");
static ZH_HANT: DataStruct = include!("zh-Hant.rs.data");
static ZH: DataStruct = include!("zh.rs.data");
