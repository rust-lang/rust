//! Checks for usage of &Vec[_] and &String
//!
//! This lint is **warn** by default

use rustc::plugin::Registry;
use rustc::lint::*;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::ptr::P;
use syntax::codemap::Span;
use types::match_ty_unwrap;

declare_lint! {
    pub VEC_PTR_ARG,
    Allow,
    "Warn on declaration of a &Vec-typed method argument"
}


#[derive(Copy,Clone)]
pub struct VecPtrArg;

impl LintPass for VecPtrArg {
    fn get_lints(&self) -> LintArray {
        lint_array!(VEC_PTR_ARG)
    }
    
    fn check_item(&mut self, cx: &Context, item: &Item) {
		if let &ItemFn(ref decl, _, _, _, _) = &item.node {
			check_fn(cx, decl);
		}
	}
    
    fn check_impl_item(&mut self, cx: &Context, item: &ImplItem) { 
		if let &MethodImplItem(ref sig, _) = &item.node {
			check_fn(cx, &sig.decl);
		}
	}
        
	fn check_trait_item(&mut self, cx: &Context, item: &TraitItem) {
		if let &MethodTraitItem(ref sig, _) = &item.node {
			check_fn(cx, &sig.decl);
		}
	}
}

fn check_fn(cx: &Context, decl: &FnDecl) {
	for arg in &decl.inputs {
		let ty = &arg.ty;
		match ty.node {
			TyPtr(ref pty) => check_ptr_subtype(cx, ty.span, &pty.ty),
			TyRptr(_, ref rpty) => check_ptr_subtype(cx, ty.span, &rpty.ty),
			_ => ()
		}
	}
}

fn check_ptr_subtype(cx: &Context, span: Span, ty: &Ty) {
	if match_ty_unwrap(ty, &["Vec"]).is_some() { 
		cx.span_lint(VEC_PTR_ARG, span, 
			"Writing '&Vec<_>' instead of '&[_]' involves one more reference and cannot be used with non-vec-based slices. Consider changing the type to &[...]");
	}
}
