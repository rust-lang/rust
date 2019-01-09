// Test that `quote`-related macro are gated by `quote` feature gate.

// (To sanity-check the code, uncomment this.)
// #![feature(quote)]

// FIXME the error message that is current emitted seems pretty bad.

// gate-test-quote

#![feature(rustc_private)]
#![allow(dead_code, unused_imports, unused_variables)]

#[macro_use]
extern crate syntax;

use syntax::ast;
use syntax::parse;

struct ParseSess;

impl ParseSess {
    fn cfg(&self) -> ast::CrateConfig { loop { } }
    fn parse_sess<'a>(&'a self) -> &'a parse::ParseSess { loop { } }
    fn call_site(&self) -> () { loop { } }
    fn ident_of(&self, st: &str) -> ast::Ident { loop { } }
    fn name_of(&self, st: &str) -> ast::Name { loop { } }
}

pub fn main() {
    let ecx = &ParseSess;
    let x = quote_tokens!(ecx, 3);
    //~^ ERROR cannot find macro `quote_tokens!` in this scope
    let x = quote_expr!(ecx, 3);
    //~^ ERROR cannot find macro `quote_expr!` in this scope
    let x = quote_ty!(ecx, 3);
    //~^ ERROR cannot find macro `quote_ty!` in this scope
    let x = quote_method!(ecx, 3);
    //~^ ERROR cannot find macro `quote_method!` in this scope
    let x = quote_item!(ecx, 3);
    //~^ ERROR cannot find macro `quote_item!` in this scope
    let x = quote_pat!(ecx, 3);
    //~^ ERROR cannot find macro `quote_pat!` in this scope
    let x = quote_arm!(ecx, 3);
    //~^ ERROR cannot find macro `quote_arm!` in this scope
    let x = quote_stmt!(ecx, 3);
    //~^ ERROR cannot find macro `quote_stmt!` in this scope
    let x = quote_attr!(ecx, 3);
    //~^ ERROR cannot find macro `quote_attr!` in this scope
    let x = quote_arg!(ecx, 3);
    //~^ ERROR cannot find macro `quote_arg!` in this scope
    let x = quote_block!(ecx, 3);
    //~^ ERROR cannot find macro `quote_block!` in this scope
    let x = quote_meta_item!(ecx, 3);
    //~^ ERROR cannot find macro `quote_meta_item!` in this scope
    let x = quote_path!(ecx, 3);
    //~^ ERROR cannot find macro `quote_path!` in this scope
}
