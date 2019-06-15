// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate rustc_plugin;
extern crate syntax_pos;
extern crate syntax;

use rustc_plugin::Registry;
use syntax_pos::Span;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::feature_gate::AttributeType;
use syntax::symbol::Symbol;
use syntax::tokenstream::TokenTree;

fn empty(_: &mut ExtCtxt, _: Span, _: &[TokenTree]) -> Box<dyn MacResult + 'static> {
    MacEager::items(Default::default())
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_attribute(Symbol::intern("foo"), AttributeType::Normal);
    reg.register_attribute(Symbol::intern("bar"), AttributeType::CrateLevel);
    reg.register_attribute(Symbol::intern("baz"), AttributeType::Whitelisted);
    reg.register_macro("empty", empty);
}
