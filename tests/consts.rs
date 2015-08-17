#![allow(plugin_as_library)]
#![feature(rustc_private)]

extern crate clippy;
extern crate syntax;
extern crate rustc;

use clippy::consts::{constant, ConstantVariant};
use clippy::consts::ConstantVariant::*;
use syntax::ast::*;
use syntax::parse::token::InternedString;
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

fn spanned<T>(t: T) -> Spanned<T> {
    Spanned{ node: t, span: COMMAND_LINE_SP }
}

fn expr(n: Expr_) -> Expr {
    Expr{
        id: 1,
        node: n,
        span: COMMAND_LINE_SP,
    }
}

fn lit(l: Lit_) -> Expr {
    expr(ExprLit(P(spanned(l))))
}

fn binop(op: BinOp_, l: Expr, r: Expr) -> Expr {
    expr(ExprBinary(spanned(op), P(l), P(r)))
}

fn check(expect: ConstantVariant, expr: &Expr) {
    assert_eq!(Some(expect), constant(ctx(), expr).map(|x| x.constant))
}

const TRUE : ConstantVariant = ConstantBool(true);
const FALSE : ConstantVariant = ConstantBool(false);
const ZERO : ConstantVariant = ConstantInt(0, UnsuffixedIntLit(Plus));

#[test]
fn test_lit() {
    check(TRUE, &lit(LitBool(true)));
    check(FALSE, &lit(LitBool(false)));
    check(ZERO, &lit(LitInt(0, UnsuffixedIntLit(Plus))));
    check(ConstantStr("cool!".into(), CookedStr), &lit(LitStr(
        InternedString::new("cool!"), CookedStr)));
}

#[test]
fn test_ops() {
    check(TRUE, &binop(BiOr, lit(LitBool(false)), lit(LitBool(true))));
    check(FALSE, &binop(BiAnd, lit(LitBool(false)), lit(LitBool(true))));

    let litzero = lit(LitInt(0, UnsuffixedIntLit(Plus)));
    check(TRUE, &binop(BiEq, litzero.clone(), litzero.clone()));
    check(TRUE, &binop(BiGe, litzero.clone(), litzero.clone()));
    check(TRUE, &binop(BiLe, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiNe, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiGt, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiLt, litzero.clone(), litzero.clone()));
}
