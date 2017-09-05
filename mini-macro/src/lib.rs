#![feature(plugin_registrar, rustc_private)]

extern crate rustc_plugin;
extern crate syntax;

use syntax::codemap::Span;
use syntax::tokenstream::TokenTree;
use syntax::ext::base::{ExtCtxt, MacEager, MacResult};
use syntax::ext::build::AstBuilder; // trait for expr_usize
use rustc_plugin::Registry;

fn expand_macro(cx: &mut ExtCtxt, sp: Span, _: &[TokenTree]) -> Box<MacResult + 'static> {
    let e = cx.expr_usize(sp, 42);
    let e = cx.expr_mut_addr_of(sp, e);
    MacEager::expr(cx.expr_mut_addr_of(sp, e))
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("mini_macro", expand_macro);
}
