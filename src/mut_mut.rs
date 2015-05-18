use syntax::ptr::P;
use syntax::ast::*;
use rustc::lint::{Context, LintPass, LintArray, Lint};
use rustc::middle::ty::{expr_ty, sty, ty_ptr, ty_rptr, mt};

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
		
		unwrap_addr(expr).map(|e| {
			if unwrap_addr(e).is_some() {
				cx.span_lint(MUT_MUT, expr.span, 
					"We're not sure what this means, so if you know, please tell us.")
			} else {
				match expr_ty(cx.tcx, e).sty {
					ty_ptr(mt{ty: _, mutbl: MutMutable}) |
					ty_rptr(_, mt{ty: _, mutbl: MutMutable}) => 
						cx.span_lint(MUT_MUT, expr.span,
							"This expression mutably borrows a mutable reference. Consider direct reborrowing"),
					_ => ()
				}
			}
		});
	}
	
	fn check_ty(&mut self, cx: &Context, ty: &Ty) {
		if unwrap_mut(ty).and_then(unwrap_mut).is_some() {
			cx.span_lint(MUT_MUT, ty.span, 
				"We're not sure what this means, so if you know, please tell us.")
		}
	}
}

fn unwrap_mut(ty : &Ty) -> Option<&Ty> {
	match ty.node {
		TyPtr(MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
		TyRptr(_, MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
		_ => Option::None
	}
}
