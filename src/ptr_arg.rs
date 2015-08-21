//! Checks for usage of &Vec[_] and &String
//!
//! This lint is **warn** by default

use rustc::lint::*;
use syntax::ast::*;
use rustc::middle::ty;

use utils::{span_lint, match_def_path};

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

#[allow(unused_imports)]
fn check_fn(cx: &Context, decl: &FnDecl) {
    {
        // In case stuff gets moved around
        use collections::vec::Vec;
        use collections::string::String;
    }
    for arg in &decl.inputs {
        if arg.ty.node == TyInfer {  // "self" arguments
            continue;
        }
        let ref sty = cx.tcx.pat_ty(&*arg.pat).sty;
        if let &ty::TyRef(_, ty::TypeAndMut { ty, mutbl: MutImmutable }) = sty {
            if let ty::TyStruct(did, _) = ty.sty {
                if match_def_path(cx, did.did, &["collections", "vec", "Vec"]) {
                    span_lint(cx, PTR_ARG, arg.ty.span,
                              "writing `&Vec<_>` instead of `&[_]` involves one more reference \
                               and cannot be used with non-Vec-based slices. Consider changing \
                               the type to `&[...]`");
                }
                else if match_def_path(cx, did.did, &["collections", "string", "String"]) {
                    span_lint(cx, PTR_ARG, arg.ty.span,
                              "writing `&String` instead of `&str` involves a new object \
                               where a slice will do. Consider changing the type to `&str`");
                }
            }
        }
    }
}
