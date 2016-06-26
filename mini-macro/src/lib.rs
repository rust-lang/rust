#![feature(plugin_registrar, rustc_private)]

extern crate syntax;
extern crate rustc;
extern crate rustc_plugin;

use syntax::codemap::Span;
use syntax::ast::TokenTree;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::ext::build::AstBuilder;  // trait for expr_usize
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
