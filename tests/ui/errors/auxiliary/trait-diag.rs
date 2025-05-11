//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: -Zremap-path-scope=diagnostics

pub trait Trait: std::fmt::Display {}
