// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
extern crate syntax;

use rustc_plugin::Registry;

use syntax::ext::base::SyntaxExtension;
use syntax::parse::token::Token;
use syntax::symbol::Symbol;
use syntax::tokenstream::{TokenTree, TokenStream};

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(Symbol::intern("cond"),
                                  SyntaxExtension::ProcMacro(Box::new(cond)));
}

fn cond(input: TokenStream) -> TokenStream {
    let mut conds = Vec::new();
    let mut input = input.trees();
    while let Some(tree) = input.next() {
        let cond: TokenStream = match *tree {
            TokenTree::Delimited(_, ref delimited) => delimited.tts.iter().cloned().collect(),
            _ => panic!("Invalid input"),
        };
        let mut trees = cond.trees().cloned();
        let test = trees.next();
        let rhs = trees.collect::<TokenStream>();
        if rhs.is_empty() {
            panic!("Invalid macro usage in cond: {}", cond);
        }
        let is_else = match test {
            Some(TokenTree::Token(_, Token::Ident(ident))) if ident.name == "else" => true,
            _ => false,
        };
        conds.push(if is_else || input.peek().is_none() {
            qquote!({ unquote rhs })
        } else {
            qquote!(if unquote(test.unwrap()) { unquote rhs } else)
        });
    }

    conds.into_iter().collect()
}
