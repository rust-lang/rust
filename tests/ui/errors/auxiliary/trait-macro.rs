//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: -Zremap-path-scope=macro

pub trait Trait: std::fmt::Display {}
