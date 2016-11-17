// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin, plugin_registrar, rustc_private)]

extern crate proc_macro_tokens;
extern crate rustc_plugin;
extern crate syntax;

use proc_macro_tokens::prelude::*;
use rustc_plugin::Registry;
use syntax::ext::base::SyntaxExtension;
use syntax::ext::proc_macro_shim::prelude::*;
use syntax::symbol::Symbol;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(Symbol::intern("attr_tru"),
                                  SyntaxExtension::AttrProcMacro(Box::new(attr_tru)));
    reg.register_syntax_extension(Symbol::intern("attr_identity"),
                                  SyntaxExtension::AttrProcMacro(Box::new(attr_identity)));
    reg.register_syntax_extension(Symbol::intern("tru"),
                                  SyntaxExtension::ProcMacro(Box::new(tru)));
    reg.register_syntax_extension(Symbol::intern("ret_tru"),
                                  SyntaxExtension::ProcMacro(Box::new(ret_tru)));
    reg.register_syntax_extension(Symbol::intern("identity"),
                                  SyntaxExtension::ProcMacro(Box::new(identity)));
}

fn attr_tru(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    lex("fn f1() -> bool { true }")
}

fn attr_identity(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let source = item.to_string();
    lex(&source)
}

fn tru(_ts: TokenStream) -> TokenStream {
    lex("true")
}

fn ret_tru(_ts: TokenStream) -> TokenStream {
    lex("return true;")
}

fn identity(ts: TokenStream) -> TokenStream {
    let source = ts.to_string();
    lex(&source)
}
