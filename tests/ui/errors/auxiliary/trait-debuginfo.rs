//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: -Zremap-path-scope=debuginfo

pub trait Trait: std::fmt::Display {}
