// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin)]
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![plugin(proc_macro_plugin)]

extern crate rustc_plugin;
extern crate syntax;

use rustc_plugin::Registry;
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;
use syntax::tokenstream::TokenStream;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(Symbol::intern("hello"),
                                  SyntaxExtension::ProcMacro(Box::new(hello)));
}

// This macro is not very interesting, but it does contain delimited tokens with
// no content - `()` and `{}` - which has caused problems in the past.
fn hello(_: TokenStream) -> TokenStream {
    qquote!({ fn hello() {} hello(); })
}
