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
use utils::span_lint;

declare_lint! {
    pub PTR_ARG,
    Allow,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` \
     instead, respectively"
}

#[derive(Copy,Clone)]
pub struct PtrArg;

impl LintPass for PtrArg {
    fn get_lints(&self) -> LintArray {
        lint_array!(PTR_ARG)
    }

    fn check_item(&mut self, cx: &Context, item: &Item) {
        if let &ItemFn(ref decl, _, _, _, _, _) = &item.node {
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
        match &arg.ty.node {
            &TyPtr(ref p) | &TyRptr(_, ref p) =>
                check_ptr_subtype(cx, arg.ty.span, &p.ty),
            _ => ()
        }
    }
}

fn check_ptr_subtype(cx: &Context, span: Span, ty: &Ty) {
    match_ty_unwrap(ty, &["Vec"]).map_or_else(|| match_ty_unwrap(ty,
        &["String"]).map_or((), |_| {
            span_lint(cx, PTR_ARG, span,
                      "writing `&String` instead of `&str` involves a new object \
                       where a slice will do. Consider changing the type to `&str`")
        }), |_| span_lint(cx, PTR_ARG, span,
                          "writing `&Vec<_>` instead of \
                           `&[_]` involves one more reference and cannot be used with \
                           non-Vec-based slices. Consider changing the type to `&[...]`"))
}
