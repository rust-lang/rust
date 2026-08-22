use rustc_data_structures::fx::FxHashSet;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, HirId};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, TyCtxt, TypeckResults};
use rustc_span::def_id::LocalDefId;
use rustc_span::{ExpnKind, MacroKind, Span};

/// Function information extracted from HIR by the coverage instrumentor.
#[derive(Debug)]
pub(crate) struct ExtractedHirInfo {
    pub(crate) function_source_hash: u64,
    pub(crate) is_async_fn: bool,
    /// The span of the function's signature, if available.
    /// Must have the same context and filename as the body span.
    pub(crate) fn_sig_span: Option<Span>,
    pub(crate) body_span: Span,
    /// "Holes" are regions within the function body (or its expansions) that
    /// should not be included in coverage spans for this function
    /// (e.g. closures and nested items).
    pub(crate) hole_spans: Vec<Span>,
    /// HIR nodes that should be ignored when extracting spans from marker
    /// statements in MIR.
    pub(crate) nodes_to_ignore: FxHashSet<HirId>,
}

pub(crate) fn extract_hir_info<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> ExtractedHirInfo {
    // FIXME(#79625): Consider improving MIR to provide the information needed, to avoid going back
    // to HIR for it.

    // Synthetic by-move coroutine bodies don't have useful HIR of their own.
    // Use the original coroutine body instead. These synthetic bodies are
    // created with a coroutine type, so we can inspect that type as-is.
    if tcx.is_synthetic_mir(def_id) {
        let effective_def_id =
            match *tcx.type_of(def_id).instantiate_identity().skip_normalization().kind() {
                ty::Coroutine(coroutine_def_id, _) => coroutine_def_id.expect_local(),
                _ => tcx.local_parent(def_id),
            };
        return extract_hir_info(tcx, effective_def_id);
    }

    let hir_node = tcx.hir_node_by_def_id(def_id);
    let fn_body_id = hir_node.body_id().expect("HIR node is a function with body");
    let hir_body = tcx.hir_body(fn_body_id);

    let maybe_fn_sig = hir_node.fn_sig();
    let is_async_fn = maybe_fn_sig.is_some_and(|fn_sig| fn_sig.header.is_async());

    let mut body_span = hir_body.value.span;

    use hir::{Closure, Expr, ExprKind, Node};
    // Unexpand a closure's body span back to the context of its declaration.
    // This helps with closure bodies that consist of just a single bang-macro,
    // and also with closure bodies produced by async desugaring.
    if let Node::Expr(&Expr { kind: ExprKind::Closure(&Closure { fn_decl_span, .. }), .. }) =
        hir_node
    {
        body_span = body_span.find_ancestor_in_same_ctxt(fn_decl_span).unwrap_or(body_span);
    }

    // The actual signature span is only used if it has the same context and
    // filename as the body, and precedes the body.
    let fn_sig_span = maybe_fn_sig.map(|fn_sig| fn_sig.span).filter(|&fn_sig_span| {
        let source_map = tcx.sess.source_map();
        let file_idx = |span: Span| source_map.lookup_source_file_idx(span.lo());

        fn_sig_span.eq_ctxt(body_span)
            && fn_sig_span.hi() <= body_span.lo()
            && file_idx(fn_sig_span) == file_idx(body_span)
    });

    let function_source_hash = hash_mir_source(tcx, hir_body);

    let hole_spans = extract_hole_spans_from_hir(tcx, hir_body);
    let nodes_to_ignore = find_nodes_to_ignore(tcx, def_id, hir_body);

    ExtractedHirInfo {
        function_source_hash,
        is_async_fn,
        fn_sig_span,
        body_span,
        hole_spans,
        nodes_to_ignore,
    }
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx hir::Body<'tcx>) -> u64 {
    let owner = hir_body.id().hir_id.owner;
    tcx.hir_owner_nodes(owner)
        .opt_hash
        .expect("hash should be present when coverage instrumentation is enabled")
        .to_smaller_hash()
        .as_u64()
}

fn extract_hole_spans_from_hir<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &hir::Body<'tcx>) -> Vec<Span> {
    struct HolesVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        hole_spans: Vec<Span>,
    }

    impl<'tcx> Visitor<'tcx> for HolesVisitor<'tcx> {
        /// We have special handling for nested items, but we still want to
        /// traverse into nested bodies of things that are not considered items,
        /// such as "anon consts" (e.g. array lengths).
        type NestedFilter = nested_filter::OnlyBodies;

        fn maybe_tcx(&mut self) -> TyCtxt<'tcx> {
            self.tcx
        }

        /// We override `visit_nested_item` instead of `visit_item` because we
        /// only need the item's span, not the item itself.
        fn visit_nested_item(&mut self, id: hir::ItemId) -> Self::Result {
            let span = self.tcx.def_span(id.owner_id.def_id);
            self.visit_hole_span(span);
            // Having visited this item, we don't care about its children,
            // so don't call `walk_item`.
        }

        // We override `visit_expr` instead of the more specific expression
        // visitors, so that we have direct access to the expression span.
        fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
            match expr.kind {
                hir::ExprKind::Closure(_) | hir::ExprKind::ConstBlock(_) => {
                    self.visit_hole_span(expr.span);
                    // Having visited this expression, we don't care about its
                    // children, so don't call `walk_expr`.
                }

                // For other expressions, recursively visit as normal.
                _ => hir::intravisit::walk_expr(self, expr),
            }
        }
    }
    impl HolesVisitor<'_> {
        fn visit_hole_span(&mut self, hole_span: Span) {
            self.hole_spans.push(hole_span);
        }
    }

    let mut visitor = HolesVisitor { tcx, hole_spans: vec![] };

    visitor.visit_body(hir_body);
    visitor.hole_spans
}

fn find_nodes_to_ignore<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    hir_body: &hir::Body<'tcx>,
) -> FxHashSet<HirId> {
    struct DivergingCallVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        typeck_results: &'tcx TypeckResults<'tcx>,
        nodes_to_ignore: FxHashSet<HirId>,
    }
    struct SubexprsVisitor<'a, 'tcx> {
        inner: &'a mut DivergingCallVisitor<'tcx>,
    }

    // Use a heuristic to detect and ignore the message-formatting arguments of assert macros,
    // because tracking coverage for them is almost never useful. This avoids big regressions
    // in coverage-report quality for code containing assertions.
    //
    // FIXME(Zalathar): Now that we have HIR-aware instrumentation, we might be able to
    // get rid of this by extending `#[coverage(off)]` to support arbitrary expressions,
    // and using it in the standard-library assert macros.
    impl<'tcx> Visitor<'tcx> for DivergingCallVisitor<'tcx> {
        fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
            // Look for call expressions with a return type of `!` produced by a bang-macro.
            // If we find one, ignore all subexpressions in the call's _arguments_.
            if let hir::ExprKind::Call(callee, args) = expr.kind
                && let callee_ty = self.typeck_results.node_type(callee.hir_id)
                && callee_ty.is_fn()
                && let Some(output) = callee_ty.fn_sig(self.tcx).output().no_bound_vars()
                && output.is_never()
                && let ExpnKind::Macro(MacroKind::Bang, _) = expr.span.ctxt().outer_expn_data().kind
            {
                for arg in args {
                    (SubexprsVisitor { inner: self }).visit_expr(arg)
                }
            } else {
                hir::intravisit::walk_expr(self, expr);
            }
        }
    }
    impl<'tcx> Visitor<'tcx> for SubexprsVisitor<'_, 'tcx> {
        fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
            self.inner.nodes_to_ignore.insert(expr.hir_id);
            hir::intravisit::walk_expr(self, expr);
        }
    }

    let mut visitor = DivergingCallVisitor {
        tcx,
        typeck_results: tcx.typeck(def_id),
        nodes_to_ignore: FxHashSet::default(),
    };
    visitor.visit_body(hir_body);
    visitor.nodes_to_ignore
}
