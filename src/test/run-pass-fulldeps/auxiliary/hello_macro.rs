// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro, proc_macro_non_items)]

extern crate proc_macro;

use proc_macro::{TokenStream, quote};

// This macro is not very interesting, but it does contain delimited tokens with
// no content - `()` and `{}` - which has caused problems in the past.
// Also, it tests that we can escape `$` via `$$`.
#[proc_macro]
pub fn hello(_: TokenStream) -> TokenStream {
    quote!({
        fn hello() {}
        macro_rules! m { ($$($$t:tt)*) => { $$($$t)* } }
        m!(hello());
    })
}
