// force-host

#![feature(plugin_registrar, quote, rustc_private)]
#![crate_type = "dylib"]

extern crate syntax;
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_plugin;
extern crate syntax_pos;

use rustc_data_structures::small_vec::OneVector;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::tokenstream;
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("multiple_items", expand)
}

fn expand(cx: &mut ExtCtxt, _: syntax_pos::Span, _: &[tokenstream::TokenTree])
          -> Box<MacResult+'static> {
    MacEager::items(OneVector::from_vec(vec![
        quote_item!(cx, struct Struct1;).unwrap(),
        quote_item!(cx, struct Struct2;).unwrap()
    ]))
}
