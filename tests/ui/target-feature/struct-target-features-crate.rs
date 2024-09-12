//@ only-x86_64
//@ aux-build: struct-target-features-crate-dep.rs
//@ check-pass
#![feature(target_feature_11)]
#![feature(struct_target_features)]
//~^ WARNING the feature `struct_target_features` is incomplete and may not be safe to use and/or cause compiler crashes

extern crate struct_target_features_crate_dep;

#[target_feature(enable = "avx")]
fn avx() {}

#[target_feature(from_args)]
fn f(_: struct_target_features_crate_dep::Avx) {
    avx();
}

fn g(_: struct_target_features_crate_dep::NoFeatures) {}

fn main() {
    if is_x86_feature_detected!("avx") {
        let avx = unsafe { struct_target_features_crate_dep::Avx {} };
        f(avx);
    }
    g(struct_target_features_crate_dep::NoFeatures {});
}
