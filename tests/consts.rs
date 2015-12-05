#![allow(plugin_as_library)]
#![feature(rustc_private)]

extern crate clippy;
extern crate syntax;
extern crate rustc;
extern crate rustc_front;

use rustc_front::hir::*;
use syntax::parse::token::InternedString;
use syntax::ptr::P;
use syntax::codemap::{Spanned, COMMAND_LINE_SP};

use syntax::ast::Lit_::*;
use syntax::ast::Lit_;
use syntax::ast::LitIntType::*;
use syntax::ast::StrStyle::*;
use syntax::ast::Sign::*;

use clippy::consts::{constant_simple, Constant};
use clippy::consts::Constant::*;

fn spanned<T>(t: T) -> Spanned<T> {
    Spanned{ node: t, span: COMMAND_LINE_SP }
}

fn expr(n: Expr_) -> Expr {
    Expr{
        id: 1,
        node: n,
        span: COMMAND_LINE_SP,
        attrs: None
    }
}

fn lit(l: Lit_) -> Expr {
    expr(ExprLit(P(spanned(l))))
}

fn binop(op: BinOp_, l: Expr, r: Expr) -> Expr {
    expr(ExprBinary(spanned(op), P(l), P(r)))
}

fn check(expect: Constant, expr: &Expr) {
    assert_eq!(Some(expect), constant_simple(expr))
}

const TRUE : Constant = ConstantBool(true);
const FALSE : Constant = ConstantBool(false);
const ZERO : Constant = ConstantInt(0, UnsuffixedIntLit(Plus));
const ONE : Constant = ConstantInt(1, UnsuffixedIntLit(Plus));
const TWO : Constant = ConstantInt(2, UnsuffixedIntLit(Plus));

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
    let litone = lit(LitInt(1, UnsuffixedIntLit(Plus)));
    check(TRUE, &binop(BiEq, litzero.clone(), litzero.clone()));
    check(TRUE, &binop(BiGe, litzero.clone(), litzero.clone()));
    check(TRUE, &binop(BiLe, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiNe, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiGt, litzero.clone(), litzero.clone()));
    check(FALSE, &binop(BiLt, litzero.clone(), litzero.clone()));

    check(ZERO, &binop(BiAdd, litzero.clone(), litzero.clone()));
    check(TWO, &binop(BiAdd, litone.clone(), litone.clone()));
    check(ONE, &binop(BiSub, litone.clone(), litzero.clone()));
    check(ONE, &binop(BiMul, litone.clone(), litone.clone()));
    check(ONE, &binop(BiDiv, litone.clone(), litone.clone()));
}
