use rustc::hir::*;
use rustc::hir::map::*;
use rustc::hir::intravisit::FnKind;
use rustc::lint::*;
use rustc::ty::TypeVariants;
use rustc_target::spec::abi::Abi;
use rustc_target::abi::LayoutOf;
use syntax::ast::NodeId;
use syntax_pos::Span;
use crate::utils::{in_macro, is_copy, is_self, span_lint_and_sugg, snippet};

/// **What it does:** Checks for functions taking arguments by reference, where
/// the argument type is `Copy` and small enough to be more efficient to always
/// pass by value.
///
/// **Why is this bad?** In many calling conventions instances of structs will
/// be passed through registers if they fit into two or less general purpose
/// registers.
///
/// **Example:**
/// ```rust
/// fn foo(v: &u32) {
///     assert_eq!(v, 42);
/// }
/// // should be
/// fn foo(v: u32) {
///     assert_eq!(v, 42);
/// }
/// ```
declare_clippy_lint! {
    pub TRIVIALLY_COPY_PASS_BY_REF,
    perf,
    "functions taking small copyable arguments by reference"
}

pub struct TriviallyCopyPassByRef;

impl LintPass for TriviallyCopyPassByRef {
    fn get_lints(&self) -> LintArray {
        lint_array![TRIVIALLY_COPY_PASS_BY_REF]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TriviallyCopyPassByRef {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        node_id: NodeId,
    ) {
        if in_macro(span) {
            return;
        }

        match kind {
            FnKind::ItemFn(.., abi, _, attrs) => {
                if abi != Abi::Rust {
                    return;
                }
                for a in attrs {
                    if a.meta_item_list().is_some() && a.name() == "proc_macro_derive" {
                        return;
                    }
                }
            },
            FnKind::Method(..) => (),
            _ => return,
        }

        // Exclude non-inherent impls
        if let Some(NodeItem(item)) = cx.tcx.hir.find(cx.tcx.hir.get_parent_node(node_id)) {
            if matches!(item.node, ItemImpl(_, _, _, _, Some(_), _, _) |
                ItemTrait(..))
            {
                return;
            }
        }

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        for ((input, &ty), arg) in decl.inputs.iter().zip(fn_sig.inputs()).zip(&body.arguments) {
            // All spans generated from a proc-macro invocation are the same...
            if span == input.span {
                return;
            }

            if_chain! {
                if let TypeVariants::TyRef(_, ty, Mutability::MutImmutable) = ty.sty;
                if is_copy(cx, ty);
                if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes());
                if size < 16;
                if let Ty_::TyRptr(_, MutTy { ty: ref decl_ty, .. }) = input.node;
                then {
                    let value_type = if is_self(arg) {
                        "self".into()
                    } else {
                        snippet(cx, decl_ty.span, "_").into()
                    };
                    span_lint_and_sugg(
                        cx,
                        TRIVIALLY_COPY_PASS_BY_REF,
                        input.span,
                        "this argument is passed by reference, but would be more efficient if passed by value",
                        "consider passing by value instead",
                        value_type);
                }
            }
        }
    }
}
