#![feature(plugin_registrar, rustc_private, quote)]

extern crate rustc_plugin;
extern crate syntax;

use rustc_plugin::Registry;
use syntax::ast::MetaItem;
use syntax::codemap::Span;
use syntax::ext::base::{Annotatable, ExtCtxt, MacEager, MacResult, SyntaxExtension};
use syntax::ext::build::AstBuilder; // trait for expr_usize
use syntax::symbol::Symbol;
use syntax::tokenstream::TokenTree;

fn expand_macro(cx: &mut ExtCtxt, sp: Span, _: &[TokenTree]) -> Box<MacResult + 'static> {
    let e = cx.expr_usize(sp, 42);
    let e = cx.expr_mut_addr_of(sp, e);
    MacEager::expr(cx.expr_mut_addr_of(sp, e))
}

fn expand_attr_macro(cx: &mut ExtCtxt, _: Span, _: &MetaItem, annotated: Annotatable) -> Vec<Annotatable> {
    vec![
        Annotatable::Item(
            quote_item!(
                cx,
                #[allow(unused)] fn needless_take_by_value(s: String) { println!("{}", s.len()); }
            ).unwrap()
        ),
        annotated,
    ]
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("mini_macro", expand_macro);
    reg.register_syntax_extension(
        Symbol::intern("mini_macro_attr"),
        SyntaxExtension::MultiModifier(Box::new(expand_attr_macro)),
    );
}
