// force-host

#![crate_type = "dylib"]
#![feature(plugin_registrar, quote, rustc_private)]

extern crate syntax_extension_with_dll_deps_1 as other;
extern crate syntax;
extern crate syntax_pos;
extern crate rustc;
extern crate rustc_plugin;

use syntax::ast::{Item, MetaItem};
use syntax::ext::base::*;
use syntax::tokenstream::TokenTree;
use syntax_pos::Span;
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("foo", expand_foo);
}

fn expand_foo(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
              -> Box<MacResult+'static> {
    let answer = other::the_answer();
    MacEager::expr(quote_expr!(cx, $answer))
}
