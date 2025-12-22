use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, ItemKind, Node, find_attr};
use rustc_middle::ty::adjustment::Adjust;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{ConstItemInteriorMutationsDiag, ConstItemInteriorMutationsSuggestionStatic};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `const_item_interior_mutations` lint checks for calls which
    /// mutates an interior mutable const-item.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::sync::Once;
    ///
    /// const INIT: Once = Once::new(); // using `INIT` will always create a temporary and
    ///                                 // never modify it-self on use, should be a `static`
    ///                                 // instead for shared use
    ///
    /// fn init() {
    ///     INIT.call_once(|| {
    ///         println!("Once::call_once first call");
    ///     });
    ///     INIT.call_once(|| {                          // this second will also print
    ///         println!("Once::call_once second call"); // as each call to `INIT` creates
    ///     });                                          // new temporary
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Calling a method which mutates an interior mutable type has no effect as const-item
    /// are essentially inlined wherever they are used, meaning that they are copied
    /// directly into the relevant context when used rendering modification through
    /// interior mutability ineffective across usage of that const-item.
    ///
    /// The current implementation of this lint only warns on significant `std` and
    /// `core` interior mutable types, like `Once`, `AtomicI32`, ... this is done out
    /// of prudence to avoid false-positive and may be extended in the future.
    pub CONST_ITEM_INTERIOR_MUTATIONS,
    Warn,
    "checks for calls which mutates a interior mutable const-item"
}

declare_lint_pass!(InteriorMutableConsts => [CONST_ITEM_INTERIOR_MUTATIONS]);

impl<'tcx> LateLintPass<'tcx> for InteriorMutableConsts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let typeck = cx.typeck_results();

        let (method_did, receiver) = match expr.kind {
            // matching on `<receiver>.method(..)`
            ExprKind::MethodCall(_, receiver, _, _) => {
                (typeck.type_dependent_def_id(expr.hir_id), receiver)
            }
            // matching on `function(&<receiver>, ...)`
            ExprKind::Call(path, [receiver, ..]) => match receiver.kind {
                ExprKind::AddrOf(_, _, receiver) => match path.kind {
                    ExprKind::Path(ref qpath) => {
                        (cx.qpath_res(qpath, path.hir_id).opt_def_id(), receiver)
                    }
                    _ => return,
                },
                _ => return,
            },
            _ => return,
        };

        let Some(method_did) = method_did else {
            return;
        };

        if let ExprKind::Path(qpath) = &receiver.kind
            && let Res::Def(DefKind::Const | DefKind::AssocConst, const_did) =
                typeck.qpath_res(qpath, receiver.hir_id)
            // Don't consider derefs as those can do arbitrary things
            // like using thread local (see rust-lang/rust#150157)
            && !cx
                .typeck_results()
                .expr_adjustments(receiver)
                .into_iter()
                .any(|adj| matches!(adj.kind, Adjust::Deref(_)))
            // Let's do the attribute check after the other checks for perf reasons
            && find_attr!(
                cx.tcx.get_all_attrs(method_did),
                AttributeKind::RustcShouldNotBeCalledOnConstItems(_)
            )
            && let Some(method_name) = cx.tcx.opt_item_ident(method_did)
            && let Some(const_name) = cx.tcx.opt_item_ident(const_did)
            && let Some(const_ty) = typeck.node_type_opt(receiver.hir_id)
        {
            // Find the local `const`-item and create the suggestion to use `static` instead
            let sugg_static = if let Some(Node::Item(const_item)) =
                cx.tcx.hir_get_if_local(const_did)
                && let ItemKind::Const(ident, _generics, _ty, _body_id) = const_item.kind
            {
                if let Some(vis_span) = const_item.vis_span.find_ancestor_inside(const_item.span)
                    && const_item.span.can_be_used_for_suggestions()
                    && vis_span.can_be_used_for_suggestions()
                {
                    Some(ConstItemInteriorMutationsSuggestionStatic::Spanful {
                        const_: const_item.vis_span.between(ident.span),
                        before: if !vis_span.is_empty() { " " } else { "" },
                    })
                } else {
                    Some(ConstItemInteriorMutationsSuggestionStatic::Spanless)
                }
            } else {
                None
            };

            cx.emit_span_lint(
                CONST_ITEM_INTERIOR_MUTATIONS,
                expr.span,
                ConstItemInteriorMutationsDiag {
                    method_name,
                    const_name,
                    const_ty,
                    receiver_span: receiver.span,
                    sugg_static,
                },
            );
        }
    }
}
