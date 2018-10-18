// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(proc_macro_quote, proc_macro_hygiene)]
extern crate proc_macro;

use proc_macro::{TokenStream, quote};

#[proc_macro_derive(ClippyMiniMacroTest)]
pub fn mini_macro(_: TokenStream) -> TokenStream {
    quote!(
        #[allow(unused)] fn needless_take_by_value(s: String) { println!("{}", s.len()); }
        #[allow(unused)] fn needless_loop(items: &[u8]) {
            for i in 0..items.len() {
                println!("{}", items[i]);
            }
        }
    )
}
