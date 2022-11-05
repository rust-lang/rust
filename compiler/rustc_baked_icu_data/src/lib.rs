//! Baked ICU data for eager translation support.
//!
#![allow(elided_lifetimes_in_paths)]

// generated with:
// ```text
// icu4x-datagen -W --pretty --fingerprint --use-separate-crates --cldr-tag latest --icuexport-tag latest \
// --format mod -l en es fr it ja pt ru tr zh-Hans zh-Hant -k list/and@1 -o src/data
// ```

mod data;

pub use data::BakedDataProvider;

pub fn baked_data_provider() -> BakedDataProvider {
    data::BakedDataProvider
}
