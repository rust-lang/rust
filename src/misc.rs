use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::visit::{FnKind};
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
use rustc::middle::ty::{self, expr_ty, ty_str, ty_ptr, ty_rptr, ty_float};
use syntax::codemap::{Span, Spanned};


use types::span_note_and_lint;

fn walk_ty<'t>(ty: ty::Ty<'t>) -> ty::Ty<'t> {
	match ty.sty {
		ty_ptr(ref tm) | ty_rptr(_, ref tm) => walk_ty(tm.ty),
		_ => ty
	}
}

/// Handles uncategorized lints
/// Currently handles linting of if-let-able matches
#[allow(missing_copy_implementations)]
pub struct MiscPass;


declare_lint!(pub SINGLE_MATCH, Warn,
              "Warn on usage of matches with a single nontrivial arm");

impl LintPass for MiscPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SINGLE_MATCH)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMatch(ref ex, ref arms, ast::MatchSource::Normal) = expr.node {
            if arms.len() == 2 {
                if arms[0].guard.is_none() && arms[1].pats.len() == 1 {
                    match arms[1].body.node {
                        ExprTup(ref v) if v.len() == 0 && arms[1].guard.is_none() => (),
                        ExprBlock(ref b) if b.stmts.len() == 0 && arms[1].guard.is_none() => (),
                         _ => return
                    }
                    // In some cases, an exhaustive match is preferred to catch situations when
                    // an enum is extended. So we only consider cases where a `_` wildcard is used
                    if arms[1].pats[0].node == PatWild(PatWildSingle) && arms[0].pats.len() == 1 {
                        let map = cx.sess().codemap();
                        span_note_and_lint(cx, SINGLE_MATCH, expr.span,
                              "You seem to be trying to use match for destructuring a single type. Did you mean to use `if let`?",
                              &*format!("Try if let {} = {} {{ ... }}",
                                      &*map.span_to_snippet(arms[0].pats[0].span).unwrap_or("..".to_string()),
                                      &*map.span_to_snippet(ex.span).unwrap_or("..".to_string()))
                        );
                    }
                }
            }
        }
    }
}


declare_lint!(pub STR_TO_STRING, Warn, "Warn when a String could use to_owned() instead of to_string()");

#[allow(missing_copy_implementations)]
pub struct StrToStringPass;

impl LintPass for StrToStringPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(STR_TO_STRING)
    }

    fn check_expr(&mut self, cx: &Context, expr: &ast::Expr) {
        match expr.node {
            ast::ExprMethodCall(ref method, _, ref args)
                if method.node.as_str() == "to_string"
                && is_str(cx, &*args[0]) => {
                cx.span_lint(STR_TO_STRING, expr.span, "str.to_owned() is faster");
            },
            _ => ()
        }

        fn is_str(cx: &Context, expr: &ast::Expr) -> bool {
            match walk_ty(expr_ty(cx.tcx, expr)).sty {
                ty_str => true,
                _ => false
            }
        }
    }
}


declare_lint!(pub TOPLEVEL_REF_ARG, Warn, "Warn about pattern matches with top-level `ref` bindings");

#[allow(missing_copy_implementations)]
pub struct TopLevelRefPass;

impl LintPass for TopLevelRefPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOPLEVEL_REF_ARG)
    }

    fn check_fn(&mut self, cx: &Context, _: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        for ref arg in decl.inputs.iter() {
            if let PatIdent(BindByRef(_), _, _) = arg.pat.node {
                cx.span_lint(
                    TOPLEVEL_REF_ARG,
                    arg.pat.span,
                    "`ref` directly on a function argument is ignored. Have you considered using a reference type instead?"
                );
            }
        }
    }
}

declare_lint!(pub CMP_NAN, Deny, "Deny comparisons to std::f32::NAN or std::f64::NAN");

#[derive(Copy,Clone)]
pub struct CmpNan;

impl LintPass for CmpNan {
	fn get_lints(&self) -> LintArray {
        lint_array!(CMP_NAN)
	}
	
	fn check_expr(&mut self, cx: &Context, expr: &Expr) {
		if let ExprBinary(ref cmp, ref left, ref right) = expr.node {
			if is_comparison_binop(cmp.node) {
				if let &ExprPath(_, ref path) = &left.node {
					check_nan(cx, path, expr.span);
				}
				if let &ExprPath(_, ref path) = &right.node {
					check_nan(cx, path, expr.span);
				}
			}
		}
	}
}

fn check_nan(cx: &Context, path: &Path, span: Span) {
	path.segments.last().map(|seg| if seg.identifier.as_str() == "NAN" {
		cx.span_lint(CMP_NAN, span, "Doomed comparison with NAN, use std::{f32,f64}::is_nan instead");
	});
}

declare_lint!(pub FLOAT_CMP, Warn,
			  "Warn on ==/!= comparison of floaty values");
			  
#[derive(Copy,Clone)]
pub struct FloatCmp;

impl LintPass for FloatCmp {
	fn get_lints(&self) -> LintArray {
        lint_array!(FLOAT_CMP)
	}
	
	fn check_expr(&mut self, cx: &Context, expr: &Expr) {
		if let ExprBinary(ref cmp, ref left, ref right) = expr.node {
			let op = cmp.node;
			if (op == BiEq || op == BiNe) && (is_float(cx, left) || is_float(cx, right)) {
				let map = cx.sess().codemap();
				cx.span_lint(FLOAT_CMP, expr.span, &format!(
					"{}-Comparison of f32 or f64 detected. You may want to change this to 'abs({} - {}) < epsilon' for some suitable value of epsilon",
					binop_to_string(op), &*map.span_to_snippet(left.span).unwrap_or("..".to_string()), 
					&*map.span_to_snippet(right.span).unwrap_or("..".to_string())));
			}
		}
	}
}

fn is_float(cx: &Context, expr: &Expr) -> bool {
	if let ty_float(_) = walk_ty(expr_ty(cx.tcx, expr)).sty { true } else { false }
}

declare_lint!(pub PRECEDENCE, Warn,
			  "Warn on mixing bit ops with integer arithmetic without parenthesis");
			  
#[derive(Copy,Clone)]
pub struct Precedence;

impl LintPass for Precedence {
	fn get_lints(&self) -> LintArray {
        lint_array!(PRECEDENCE)
	}
	
	fn check_expr(&mut self, cx: &Context, expr: &Expr) {
		if let ExprBinary(Spanned { node: op, ..}, ref left, ref right) = expr.node {
			if is_bit_op(op) && (is_arith_expr(left) || is_arith_expr(right)) {
				cx.span_lint(PRECEDENCE, expr.span, 
					"Operator precedence can trip the unwary. Consider adding parenthesis to the subexpression.");
			}
		}
	}
}

fn is_arith_expr(expr : &Expr) -> bool {
	match expr.node {
		ExprBinary(Spanned { node: op, ..}, _, _) => is_arith_op(op),
		_ => false
	}
}

fn is_bit_op(op : BinOp_) -> bool {
	match op {
		BiBitXor | BiBitAnd | BiBitOr | BiShl | BiShr => true,
		_ => false
	}
}

fn is_arith_op(op : BinOp_) -> bool {
	match op {
		BiAdd | BiSub | BiMul | BiDiv | BiRem => true,
		_ => false
	}
}
