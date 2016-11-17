// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_parens)]
#![feature(plugin)]
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![plugin(proc_macro_plugin)]

extern crate rustc_plugin;
extern crate proc_macro_tokens;
extern crate syntax;

use proc_macro_tokens::build::ident_eq;

use syntax::ast::Ident;
use syntax::ext::base::{ExtCtxt, MacResult};
use syntax::ext::proc_macro_shim::build_block_emitter;
use syntax::tokenstream::{TokenTree, TokenStream};
use syntax::codemap::Span;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("cond", cond);
}

fn cond<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> Box<MacResult + 'cx> {
    let output = cond_rec(TokenStream::from_tts(tts.clone().to_owned()));
    build_block_emitter(cx, sp, output)
}

fn cond_rec(input: TokenStream) -> TokenStream {
  if input.is_empty() {
      return qquote!();
  }

  let next = input.slice(0..1);
  let rest = input.slice_from(1..);

  let clause : TokenStream = match next.maybe_delimited() {
    Some(ts) => ts,
    _ => panic!("Invalid input"),
  };

  // clause is ([test]) [rhs]
  if clause.len() < 2 { panic!("Invalid macro usage in cond: {:?}", clause) }

  let test: TokenStream = clause.slice(0..1);
  let rhs: TokenStream = clause.slice_from(1..);

  if ident_eq(&test[0], Ident::from_str("else")) || rest.is_empty() {
    qquote!({unquote(rhs)})
  } else {
    qquote!({if unquote(test) { unquote(rhs) } else { cond!(unquote(rest)) } })
  }
}
