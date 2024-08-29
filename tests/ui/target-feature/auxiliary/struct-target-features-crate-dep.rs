#![feature(struct_target_features)]

#[target_feature(enable = "avx")]
pub struct Avx {}

pub struct NoFeatures {}
