//! Checks for usage of &Vec[_] and &String
//!
//! This lint is **warn** by default

use rustc::front::map::NodeItem;
use rustc::lint::*;
use rustc::middle::ty;
use rustc_front::hir::*;
use syntax::ast::NodeId;
use utils::{STRING_PATH, VEC_PATH};
use utils::{span_lint, match_type};

/// **What it does:** This lint checks for function arguments of type `&String` or `&Vec` unless the references are mutable.
///
/// **Why is this bad?** Requiring the argument to be of the specific size makes the function less useful for no benefit; slices in the form of `&[T]` or `&str` usually suffice and can be obtained from other types, too.
///
/// **Known problems:** None
///
/// **Example:** `fn foo(&Vec<u32>) { .. }`
declare_lint! {
    pub PTR_ARG,
    Warn,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` \
     instead, respectively"
}

#[derive(Copy,Clone)]
pub struct PtrArg;

impl LintPass for PtrArg {
    fn get_lints(&self) -> LintArray {
        lint_array!(PTR_ARG)
    }
}

impl LateLintPass for PtrArg {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemFn(ref decl, _, _, _, _, _) = item.node {
            check_fn(cx, decl, item.id);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext, item: &ImplItem) {
        if let ImplItemKind::Method(ref sig, _) = item.node {
            if let Some(NodeItem(it)) = cx.tcx.map.find(cx.tcx.map.get_parent(item.id)) {
                if let ItemImpl(_, _, _, Some(_), _, _) = it.node {
                    return; // ignore trait impls
                }
            }
            check_fn(cx, &sig.decl, item.id);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &TraitItem) {
        if let MethodTraitItem(ref sig, _) = item.node {
            check_fn(cx, &sig.decl, item.id);
        }
    }
}

fn check_fn(cx: &LateContext, decl: &FnDecl, fn_id: NodeId) {
    let fn_ty = cx.tcx.node_id_to_type(fn_id).fn_sig().skip_binder();

    for (arg, ty) in decl.inputs.iter().zip(&fn_ty.inputs) {
        if let ty::TyRef(_, ty::TypeAndMut { ty, mutbl: MutImmutable }) = ty.sty {
            if match_type(cx, ty, &VEC_PATH) {
                span_lint(cx,
                          PTR_ARG,
                          arg.ty.span,
                          "writing `&Vec<_>` instead of `&[_]` involves one more reference and cannot be used \
                           with non-Vec-based slices. Consider changing the type to `&[...]`");
            } else if match_type(cx, ty, &STRING_PATH) {
                span_lint(cx,
                          PTR_ARG,
                          arg.ty.span,
                          "writing `&String` instead of `&str` involves a new object where a slice will do. \
                           Consider changing the type to `&str`");
            }
        }
    }
}
