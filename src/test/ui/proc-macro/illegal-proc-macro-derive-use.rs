// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate proc_macro;

#[proc_macro_derive(Foo)]
//~^ ERROR: only usable with crates of the `proc-macro` crate type
pub fn foo(a: proc_macro::TokenStream) -> proc_macro::TokenStream {
    a
}

// Issue #37590
#[proc_macro_derive(Foo)]
//~^ ERROR: the `#[proc_macro_derive]` attribute may only be used on bare functions
pub struct Foo {
}

fn main() {}
