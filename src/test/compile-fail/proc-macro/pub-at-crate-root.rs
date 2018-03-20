// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "proc-macro"]

extern crate proc_macro;

pub mod a { //~ `proc-macro` crate types cannot export any items
    use proc_macro::TokenStream;

    #[proc_macro_derive(B)]
    pub fn bar(a: TokenStream) -> TokenStream {
    //~^ ERROR: must currently reside in the root of the crate
        a
    }
}

#[proc_macro_derive(B)]
fn bar(a: proc_macro::TokenStream) -> proc_macro::TokenStream {
//~^ ERROR: functions tagged with `#[proc_macro_derive]` must be `pub`
    a
}
