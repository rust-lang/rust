// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
// compile-flags:--crate-type proc-macro

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(ParsingDerive)]
pub fn parsing_derive(input: TokenStream) -> TokenStream {
    let _ = input.to_string().parse::<TokenStream>().unwrap();
    "".parse().unwrap()
}
