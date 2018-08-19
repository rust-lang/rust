// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure `$crate` and `crate` work in for basic cases of nested macros.

// compile-pass
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
