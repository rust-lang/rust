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

#![feature(proc_macro, proc_macro_non_items)]
#![crate_type = "proc-macro"]

extern crate proc_macro as proc_macro_renamed; // This does not break `quote!`

use proc_macro_renamed::{TokenStream, quote};

#[proc_macro]
pub fn hello(input: TokenStream) -> TokenStream {
    quote!(hello_helper!($input))
    //^ `hello_helper!` always resolves to the following proc macro,
    //| no matter where `hello!` is used.
}

#[proc_macro]
pub fn hello_helper(input: TokenStream) -> TokenStream {
    quote! {
        extern crate hygiene_example; // This is never a conflict error
        let string = format!("hello {}", $input);
        //^ `format!` always resolves to the prelude macro,
        //| even if a different `format!` is in scope where `hello!` is used.
        hygiene_example::print(&string)
    }
}
