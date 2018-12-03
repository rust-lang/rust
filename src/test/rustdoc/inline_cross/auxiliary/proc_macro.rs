// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// no-prefer-dynamic

#![crate_type="proc-macro"]
#![crate_name="some_macros"]

extern crate proc_macro;

use proc_macro::TokenStream;

/// a proc-macro that swallows its input and does nothing.
#[proc_macro]
pub fn some_proc_macro(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

/// a proc-macro attribute that passes its item through verbatim.
#[proc_macro_attribute]
pub fn some_proc_attr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// a derive attribute that adds nothing to its input.
#[proc_macro_derive(SomeDerive)]
pub fn some_derive(_item: TokenStream) -> TokenStream {
    TokenStream::new()
}

