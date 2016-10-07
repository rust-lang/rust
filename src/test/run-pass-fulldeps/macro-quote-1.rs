// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1

#![feature(plugin)]
#![feature(rustc_private)]
#![plugin(proc_macro_plugin)]

extern crate proc_macro_plugin;
use proc_macro_plugin::prelude::*;

extern crate syntax;
use syntax::ast::Ident;
use syntax::codemap::DUMMY_SP;
use syntax::parse::token::{self, Token, keywords, str_to_ident};

fn main() {
    let lex_true = lex("true");
    assert_eq!(qquote!(true).eq_unspanned(&lex_true), true);
}
