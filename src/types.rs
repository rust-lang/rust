use rustc::lint::*;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::ptr::P;
use rustc::middle::ty;
use syntax::codemap::ExpnInfo;

use utils::{in_macro, snippet, span_lint, span_help_and_lint, in_external_macro};

/// Handles all the linting of funky types
#[allow(missing_copy_implementations)]
pub struct TypePass;

declare_lint!(pub BOX_VEC, Warn,
              "usage of `Box<Vec<T>>`, vector elements are already on the heap");
declare_lint!(pub LINKEDLIST, Warn,
              "usage of LinkedList, usually a vector is faster, or a more specialized data \
               structure like a RingBuf");

/// Matches a type with a provided string, and returns its type parameters if successful
pub fn match_ty_unwrap<'a>(ty: &'a Ty, segments: &[&str]) -> Option<&'a [P<Ty>]> {
    match ty.node {
        TyPath(_, Path {segments: ref seg, ..}) => {
            // So ast::Path isn't the full path, just the tokens that were provided.
            // I could muck around with the maps and find the full path
            // however the more efficient way is to simply reverse the iterators and zip them
            // which will compare them in reverse until one of them runs out of segments
            if seg.iter().rev().zip(segments.iter().rev()).all(|(a,b)| a.identifier.name == b) {
                match seg[..].last() {
                    Some(&PathSegment {parameters: AngleBracketedParameters(ref a), ..}) => {
                        Some(&a.types[..])
                    }
                    _ => None
                }
            } else {
                None
            }
        },
        _ => None
    }
}

#[allow(unused_imports)]
impl LintPass for TypePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_VEC, LINKEDLIST)
    }

    fn check_ty(&mut self, cx: &Context, ty: &ast::Ty) {
        {
            // In case stuff gets moved around
            use std::boxed::Box;
            use std::vec::Vec;
        }
        match_ty_unwrap(ty, &["std", "boxed", "Box"]).and_then(|t| t.first())
          .and_then(|t| match_ty_unwrap(&**t, &["std", "vec", "Vec"]))
          .map(|_| {
            span_help_and_lint(cx, BOX_VEC, ty.span,
                              "you seem to be trying to use `Box<Vec<T>>`. Did you mean to use `Vec<T>`?",
                              "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation");
          });
        {
            // In case stuff gets moved around
            use collections::linked_list::LinkedList as DL1;
            use std::collections::linked_list::LinkedList as DL2;
        }
        let dlists = [vec!["std","collections","linked_list","LinkedList"],
                      vec!["collections","linked_list","LinkedList"]];
        for path in &dlists {
            if match_ty_unwrap(ty, &path[..]).is_some() {
                span_help_and_lint(cx, LINKEDLIST, ty.span,
                                   "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                                   "a RingBuf might work");
                return;
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
declare_lint!(pub CAST_POSSIBLE_OVERFLOW, Allow,
              "casts that may cause overflow, e.g `x as u8` where `x: u32`, or `x as i32` where `x: f32`");

impl LintPass for CastPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CAST_PRECISION_LOSS,
                    CAST_SIGN_LOSS,
                    CAST_POSSIBLE_OVERFLOW)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprCast(ref ex, _) = expr.node {
            let (cast_from, cast_to) = (cx.tcx.expr_ty(&*ex), cx.tcx.expr_ty(expr));
            if cast_from.is_numeric() && !in_external_macro(cx, expr.span) {
                match (cast_from.is_integral(), cast_to.is_integral()) {
                    (true, false)  => {
                        match (&cast_from.sty, &cast_to.sty) {
                            (&ty::TypeVariants::TyInt(i), &ty::TypeVariants::TyFloat(f)) => {
                                match (i, f) {
                                    (ast::IntTy::TyI32, ast::FloatTy::TyF32) |
                                    (ast::IntTy::TyI64, ast::FloatTy::TyF32) |
                                    (ast::IntTy::TyI64, ast::FloatTy::TyF64) => {
                                        span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                                  &format!("converting from {} to {}, which causes a loss of precision",
                                                           i, f));
                                    },
                                    _ => ()
                                }
                            }
                            (&ty::TypeVariants::TyUint(u), &ty::TypeVariants::TyFloat(f)) => {
                                match (u, f) {
                                    (ast::UintTy::TyU32, ast::FloatTy::TyF32) |
                                    (ast::UintTy::TyU64, ast::FloatTy::TyF32) |
                                    (ast::UintTy::TyU64, ast::FloatTy::TyF64) => {
                                        span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                                  &format!("converting from {} to {}, which causes a loss of precision",
                                                           u, f));
                                    },
                                    _ => ()
                                }
                            },
                            _ => ()
                        }
                    },
                    (false, true) => {
                        span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, 
                                  &format!("the contents of a {} may overflow a {}", cast_from, cast_to));
                        if !cx.tcx.expr_ty(expr).is_signed() {
                            span_lint(cx, CAST_SIGN_LOSS, expr.span, 
                                      &format!("casting from {} to {} loses the sign of the value", cast_from, cast_to));
                        }
                    },
                    (true, true) => {
                        match (&cast_from.sty, &cast_to.sty) {
                            (&ty::TypeVariants::TyInt(i1), &ty::TypeVariants::TyInt(i2)) => {
                                match (i1, i2) {
                                    (ast::IntTy::TyI64, ast::IntTy::TyI32) |
                                    (ast::IntTy::TyI64, ast::IntTy::TyI16) |
                                    (ast::IntTy::TyI64, ast::IntTy::TyI8)  |
                                    (ast::IntTy::TyI32, ast::IntTy::TyI16) |
                                    (ast::IntTy::TyI32, ast::IntTy::TyI8)  |
                                    (ast::IntTy::TyI16, ast::IntTy::TyI8) => 
                                        span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, 
                                                  &format!("the contents of a {} may overflow a {}", i1, i2)),
                                    _ => ()
                                }
                            },
                            (&ty::TypeVariants::TyInt(i), &ty::TypeVariants::TyUint(u)) => {
                                span_lint(cx, CAST_SIGN_LOSS, expr.span, 
                                          &format!("casting from {} to {} loses the sign of the value", i, u));
                                match (i, u) {
                                    (ast::IntTy::TyI64, ast::UintTy::TyU32) |
                                    (ast::IntTy::TyI64, ast::UintTy::TyU16) |
                                    (ast::IntTy::TyI64, ast::UintTy::TyU8)  |
                                    (ast::IntTy::TyI32, ast::UintTy::TyU16) |
                                    (ast::IntTy::TyI32, ast::UintTy::TyU8)  |
                                    (ast::IntTy::TyI16, ast::UintTy::TyU8) => 
                                        span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, 
                                                  &format!("the contents of a {} may overflow a {}", i, u)),
                                    _ => ()
                                }
                            },
                            (&ty::TypeVariants::TyUint(u), &ty::TypeVariants::TyInt(i)) => {
                                match (u, i) {
                                    (ast::UintTy::TyU64, ast::IntTy::TyI32) |
                                    (ast::UintTy::TyU64, ast::IntTy::TyI64) |
                                    (ast::UintTy::TyU64, ast::IntTy::TyI16) |
                                    (ast::UintTy::TyU64, ast::IntTy::TyI8)  |
                                    (ast::UintTy::TyU32, ast::IntTy::TyI32) |
                                    (ast::UintTy::TyU32, ast::IntTy::TyI16) |
                                    (ast::UintTy::TyU32, ast::IntTy::TyI8)  |
                                    (ast::UintTy::TyU16, ast::IntTy::TyI16) |
                                    (ast::UintTy::TyU16, ast::IntTy::TyI8)  |
                                    (ast::UintTy::TyU8, ast::IntTy::TyI8) => 
                                        span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, 
                                                  &format!("the contents of a {} may overflow a {}", u, i)),
                                    _ => ()
                                }
                            },
                            (&ty::TypeVariants::TyUint(u1), &ty::TypeVariants::TyUint(u2)) => {
                                match (u1, u2) {
                                    (ast::UintTy::TyU64, ast::UintTy::TyU32) |
                                    (ast::UintTy::TyU64, ast::UintTy::TyU16) |
                                    (ast::UintTy::TyU64, ast::UintTy::TyU8)  |
                                    (ast::UintTy::TyU32, ast::UintTy::TyU16) |
                                    (ast::UintTy::TyU32, ast::UintTy::TyU8)  |
                                    (ast::UintTy::TyU16, ast::UintTy::TyU8) => 
                                        span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, 
                                                  &format!("the contents of a {} may overflow a {}", u1, u2)),
                                    _ => ()
                                }
                            },
                            _ => ()
                        }
                    }
                    (false, false) => {
                        if let (&ty::TypeVariants::TyFloat(ast::FloatTy::TyF64),
                                &ty::TypeVariants::TyFloat(ast::FloatTy::TyF32)) = (&cast_from.sty, &cast_to.sty) {
                            span_lint(cx, CAST_POSSIBLE_OVERFLOW, expr.span, "the contents of a f64 may overflow a f32");
                        }
                    }
                }
            }
        }
    }                                        
}
