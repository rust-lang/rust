//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: --remap-path-scope=documentation -Zunstable-options

pub trait Trait: std::fmt::Display {}
