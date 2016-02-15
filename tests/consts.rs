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

use syntax::ast::LitKind;
use syntax::ast::LitIntType;
use syntax::ast::StrStyle;

use clippy::consts::{constant_simple, Constant, FloatWidth, Sign};

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

fn lit(l: LitKind) -> Expr {
    expr(ExprLit(P(spanned(l))))
}

fn binop(op: BinOp_, l: Expr, r: Expr) -> Expr {
    expr(ExprBinary(spanned(op), P(l), P(r)))
}

fn check(expect: Constant, expr: &Expr) {
    assert_eq!(Some(expect), constant_simple(expr))
}

const TRUE : Constant = Constant::Bool(true);
const FALSE : Constant = Constant::Bool(false);
const ZERO : Constant = Constant::Int(0, LitIntType::Unsuffixed, Sign::Plus);
const ONE : Constant = Constant::Int(1, LitIntType::Unsuffixed, Sign::Plus);
const TWO : Constant = Constant::Int(2, LitIntType::Unsuffixed, Sign::Plus);

#[test]
fn test_lit() {
    check(TRUE, &lit(LitKind::Bool(true)));
    check(FALSE, &lit(LitKind::Bool(false)));
    check(ZERO, &lit(LitKind::Int(0, LitIntType::Unsuffixed)));
    check(Constant::Str("cool!".into(), StrStyle::Cooked), &lit(LitKind::Str(
        InternedString::new("cool!"), StrStyle::Cooked)));
}

#[test]
fn test_ops() {
    check(TRUE, &binop(BiOr, lit(LitKind::Bool(false)), lit(LitKind::Bool(true))));
    check(FALSE, &binop(BiAnd, lit(LitKind::Bool(false)), lit(LitKind::Bool(true))));

    let litzero = lit(LitKind::Int(0, LitIntType::Unsuffixed));
    let litone = lit(LitKind::Int(1, LitIntType::Unsuffixed));
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

    let half_any = Constant::Float("0.5".into(), FloatWidth::Any);
    let half32 = Constant::Float("0.5".into(), FloatWidth::F32);
    let half64 = Constant::Float("0.5".into(), FloatWidth::F64);

    assert_eq!(half_any, half32);
    assert_eq!(half_any, half64);
    assert_eq!(half32, half64); // for transitivity
}
