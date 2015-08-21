use rustc::lint::*;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use rustc::middle::ty;
use syntax::codemap::ExpnInfo;

use utils::{in_macro, match_def_path, snippet, span_lint, span_help_and_lint, in_external_macro};
use utils::{LL_PATH, VEC_PATH};

/// Handles all the linting of funky types
#[allow(missing_copy_implementations)]
pub struct TypePass;

declare_lint!(pub BOX_VEC, Warn,
              "usage of `Box<Vec<T>>`, vector elements are already on the heap");
declare_lint!(pub LINKEDLIST, Warn,
              "usage of LinkedList, usually a vector is faster, or a more specialized data \
               structure like a RingBuf");

impl LintPass for TypePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_VEC, LINKEDLIST)
    }

    fn check_ty(&mut self, cx: &Context, ast_ty: &ast::Ty) {
        if let Some(ty) = cx.tcx.ast_ty_to_ty_cache.borrow().get(&ast_ty.id) {
            if let ty::TyBox(ref inner) = ty.sty {
                if let ty::TyStruct(did, _) = inner.sty {
                    if match_def_path(cx, did.did, &VEC_PATH) {
                        span_help_and_lint(
                            cx, BOX_VEC, ast_ty.span,
                            "you seem to be trying to use `Box<Vec<T>>`. Did you mean to use `Vec<T>`?",
                            "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation");
                    }
                }
            }
            if let ty::TyStruct(did, _) = ty.sty {
                if match_def_path(cx, did.did, &LL_PATH) {
                    span_help_and_lint(
                        cx, LINKEDLIST, ast_ty.span,
                        "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                        "a RingBuf might work");
                    return;
                }
            }
        }
    }
}

#[allow(missing_copy_implementations)]
pub struct LetPass;

declare_lint!(pub LET_UNIT_VALUE, Warn,
              "creating a let binding to a value of unit type, which usually can't be used afterwards");


fn check_let_unit(cx: &Context, decl: &Decl, info: Option<&ExpnInfo>) {
    if in_macro(cx, info) { return; }
    if let DeclLocal(ref local) = decl.node {
        let bindtype = &cx.tcx.pat_ty(&*local.pat).sty;
        if *bindtype == ty::TyTuple(vec![]) {
            span_lint(cx, LET_UNIT_VALUE, decl.span, &format!(
                "this let-binding has unit value. Consider omitting `let {} =`",
                snippet(cx, local.pat.span, "..")));
        }
    }
}

impl LintPass for LetPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(LET_UNIT_VALUE)
    }

    fn check_decl(&mut self, cx: &Context, decl: &Decl) {
        cx.sess().codemap().with_expn_info(
            decl.span.expn_id,
            |info| check_let_unit(cx, decl, info));
    }
}

declare_lint!(pub UNIT_CMP, Warn,
              "comparing unit values (which is always `true` or `false`, respectively)");

#[allow(missing_copy_implementations)]
pub struct UnitCmp;

impl LintPass for UnitCmp {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNIT_CMP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprBinary(ref cmp, ref left, _) = expr.node {
            let op = cmp.node;
            let sty = &cx.tcx.expr_ty(left).sty;
            if *sty == ty::TyTuple(vec![]) && is_comparison_binop(op) {
                let result = match op {
                    BiEq | BiLe | BiGe => "true",
                    _ => "false"
                };
                span_lint(cx, UNIT_CMP, expr.span, &format!(
                    "{}-comparison of unit values detected. This will always be {}",
                    binop_to_string(op), result));
            }
        }
    }
}

pub struct CastPass;

declare_lint!(pub CAST_PRECISION_LOSS, Allow,
              "casts that cause loss of precision, e.g `x as f32` where `x: u64`");
declare_lint!(pub CAST_SIGN_LOSS, Allow,
              "casts from signed types to unsigned types, e.g `x as u32` where `x: i32`");
declare_lint!(pub CAST_POSSIBLE_TRUNCATION, Allow,
              "casts that may cause truncation of the value, e.g `x as u8` where `x: u32`, or `x as i32` where `x: f32`");

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
fn int_ty_to_nbits(typ: &ty::TyS) -> usize {
    let n = match &typ.sty {
    &ty::TyInt(i) =>  4 << (i as usize),
    &ty::TyUint(u) => 4 << (u as usize),
    _ => 0
    };
    // n == 4 is the usize/isize case
    if n == 4 { ::std::usize::BITS } else { n }
}

impl LintPass for CastPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CAST_PRECISION_LOSS,
                    CAST_SIGN_LOSS,
                    CAST_POSSIBLE_TRUNCATION)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprCast(ref ex, _) = expr.node {
            let (cast_from, cast_to) = (cx.tcx.expr_ty(&*ex), cx.tcx.expr_ty(expr));
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx, expr.span) {
                match (cast_from.is_integral(), cast_to.is_integral()) {
                    (true, false) => {
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits : usize = match &cast_to.sty {
                            &ty::TyFloat(ast::TyF32) => 32,
                            &ty::TyFloat(ast::TyF64) => 64,
                            _ => 0
                        };
                        if from_nbits != 0 {
                            if from_nbits >= to_nbits {
                                span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                          &format!("converting from {0} to {1}, which causes a loss of precision \
                                          			({0} is {2} bits wide, but {1}'s mantissa is only {3} bits wide)",
                                                   cast_from, cast_to, from_nbits, if to_nbits == 64 {52} else {23} ));
                            }
                        }
                    },
                    (false, true) => {
                        span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                  &format!("casting {} to {} may cause truncation of the value", cast_from, cast_to));
                        if !cast_to.is_signed() {
                            span_lint(cx, CAST_SIGN_LOSS, expr.span,
                                      &format!("casting from {} to {} loses the sign of the value", cast_from, cast_to));
                        }
                    },
                    (true, true) => {
                        if cast_from.is_signed() && !cast_to.is_signed() {
                            span_lint(cx, CAST_SIGN_LOSS, expr.span,
                                      &format!("casting from {} to {} loses the sign of the value", cast_from, cast_to));
                        }
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits   = int_ty_to_nbits(cast_to);
                        if to_nbits < from_nbits ||
                           (!cast_from.is_signed() && cast_to.is_signed() && to_nbits <= from_nbits) {
                                span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                          &format!("casting {} to {} may cause truncation of the value", cast_from, cast_to));
                        }
                    }
                    (false, false) => {
                        if let (&ty::TyFloat(ast::TyF64),
                                &ty::TyFloat(ast::TyF32)) = (&cast_from.sty, &cast_to.sty) {
                            span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span, "casting f64 to f32 may cause truncation of the value");
                        }
                    }
                }
            }
        }
    }
}
