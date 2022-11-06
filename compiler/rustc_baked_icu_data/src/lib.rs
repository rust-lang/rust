//! Baked ICU data for eager translation support.
//!
#![allow(elided_lifetimes_in_paths)]

/* generated with:
```text
icu4x-datagen -W --pretty --fingerprint --use-separate-crates --cldr-tag latest --icuexport-tag latest \
--format mod -l en es fr it ja pt ru tr zh zh-Hans zh-Hant -k list/and@1 fallback/likelysubtags@1 fallback/parents@1 fallback/supplement/co@1 \
-o src/data
```
*/

// FIXME: Workaround https://github.com/unicode-org/icu4x/issues/2815
mod data {
    /*
    use super::data::BakedDataProvider;
    use icu_provider::{
        AnyPayload, AnyProvider, AnyResponse, DataError, DataErrorKind, DataKey, DataRequest,
        KeyedDataMarker,
    };

    use crate::data::fallback;
    use crate::data::list;
     */

    include!("data/mod.rs");
    include!("data/any.rs");
}

pub use data::BakedDataProvider;

pub const fn baked_data_provider() -> BakedDataProvider {
    data::BakedDataProvider
}

pub mod supported_locales {
    pub const EN: icu_locid::Locale = icu_locid::locale!("en");
    pub const ES: icu_locid::Locale = icu_locid::locale!("es");
    pub const FR: icu_locid::Locale = icu_locid::locale!("fr");
    pub const IT: icu_locid::Locale = icu_locid::locale!("it");
    pub const JA: icu_locid::Locale = icu_locid::locale!("ja");
    pub const PT: icu_locid::Locale = icu_locid::locale!("pt");
    pub const RU: icu_locid::Locale = icu_locid::locale!("ru");
    pub const TR: icu_locid::Locale = icu_locid::locale!("tr");
    pub const ZH_HANS: icu_locid::Locale = icu_locid::locale!("zh-Hans");
    pub const ZH_HANT: icu_locid::Locale = icu_locid::locale!("zh-Hant");
}
