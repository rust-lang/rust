#![allow(plugin_as_library)]
#![feature(rustc_private)]

extern crate clippy;
extern crate syntax;
extern crate rustc;

use clippy::consts::{constant, ConstantVariant};
use clippy::consts::ConstantVariant::*;
use syntax::ast::*;
use syntax::ptr::P;
use syntax::codemap::{Spanned, COMMAND_LINE_SP};
use std::mem;
use rustc::lint::Context;

fn ctx() -> &'static Context<'static, 'static> {
    unsafe {
        let x : *const Context<'static, 'static> = std::ptr::null();
        mem::transmute(x)
    }
}

fn lit(l: Lit_) -> Expr {
    Expr{
        id: 1,
        node: ExprLit(P(Spanned{
            node: l,
            span: COMMAND_LINE_SP,
        })),
        span: COMMAND_LINE_SP,
    }
}

fn check(expect: ConstantVariant, expr: &Expr) {
    assert_eq!(Some(expect), constant(ctx(), expr).map(|x| x.constant))
}

#[test]
fn test_lit() {
    check(ConstantBool(true), &lit(LitBool(true)));
    check(ConstantBool(false), &lit(LitBool(false)));

}
