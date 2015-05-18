use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::visit::{FnKind};
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
use rustc::middle::ty::{self, expr_ty, ty_str, ty_ptr, ty_rptr, ty_float};
use syntax::codemap::{Span, Spanned};

declare_lint!(pub MUT_MUT, Warn,
              "Warn on usage of double-mut refs, e.g. '&mut &mut ...'");

#[derive(Copy,Clone)]
pub struct MutMut;

impl LintPass for MutMut {
	fn get_lints(&self) -> LintArray {
        lint_array!(MUT_MUT)
	}
	
	fn check_expr(&mut self, cx: &Context, expr: &Expr) {
		
		fn unwrap_addr(expr : &Expr) -> Option<&Expr> {
			match expr.node {
				ExprAddrOf(MutMutable, ref e) => Option::Some(e),
				_ => Option::None
			}
		}
		
		if unwrap_addr(expr).and_then(unwrap_addr).is_some() {
			cx.span_lint(MUT_MUT, expr.span, 
				"We're not sure what this means, so if you know, please tell us.")
		}
	}
	
	fn check_ty(&mut self, cx: &Context, ty: &Ty) {
		
		fn unwrap_mut(ty : &Ty) -> Option<&Ty> {
			match ty.node {
				TyPtr(MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
				TyRptr(_, MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
				_ => Option::None
			}
		}
		
		if unwrap_mut(ty).and_then(unwrap_mut).is_some() {
			cx.span_lint(MUT_MUT, ty.span, 
				"We're not sure what this means, so if you know, please tell us.")
		}
	}
}
