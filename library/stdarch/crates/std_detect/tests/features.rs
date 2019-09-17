#![feature(stdsimd)]
#![feature(stdsimd_internal)]

extern crate std_detect;

use std_detect::detect::{features, Feature};

#[test]
fn features_roundtrip() {
    for (f, _) in features() {
        let _ = Feature::from_str(f).unwrap();
    }
}
