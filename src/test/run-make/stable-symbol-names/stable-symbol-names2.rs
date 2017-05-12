// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="rlib"]

extern crate stable_symbol_names1;

pub fn user() {
  stable_symbol_names1::generic_function(1u32);
  stable_symbol_names1::generic_function("def");
  let x = 2u64;
  stable_symbol_names1::generic_function(&x);
  stable_symbol_names1::mono_function();
  stable_symbol_names1::mono_function_lifetime(&0);
}

pub fn trait_impl_test_function() {
  use stable_symbol_names1::*;
  Bar::generic_method::<Bar>();
}

