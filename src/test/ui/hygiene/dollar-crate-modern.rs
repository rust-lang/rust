// Make sure `$crate` and `crate` work in for basic cases of nested macros.

// build-pass (FIXME(62277): could be check-pass?)
// aux-build:intercrate.rs

#![feature(decl_macro, crate_in_paths)]

extern crate intercrate;

// `$crate`
intercrate::uses_dollar_crate_modern!();

intercrate::define_uses_dollar_crate_modern_nested!(uses_dollar_crate_modern_nested);
uses_dollar_crate_modern_nested!();

intercrate::define_uses_dollar_crate_legacy_nested!();
uses_dollar_crate_legacy_nested!();

// `crate`
intercrate::uses_crate_modern!();

intercrate::define_uses_crate_modern_nested!(uses_crate_modern_nested);
uses_crate_modern_nested!();

fn main() {}
