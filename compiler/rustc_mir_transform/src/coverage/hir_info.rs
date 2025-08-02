use rustc_hir as hir;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

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
}

pub(crate) fn extract_hir_info<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> ExtractedHirInfo {
    // FIXME(#79625): Consider improving MIR to provide the information needed, to avoid going back
    // to HIR for it.

    // HACK: For synthetic MIR bodies (async closures), use the def id of the HIR body.
    if tcx.is_synthetic_mir(def_id) {
        return extract_hir_info(tcx, tcx.local_parent(def_id));
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

    ExtractedHirInfo { function_source_hash, is_async_fn, fn_sig_span, body_span, hole_spans }
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx hir::Body<'tcx>) -> u64 {
    let owner = hir_body.id().hir_id.owner;
    tcx.hir_owner_nodes(owner)
        .opt_hash_including_bodies
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
                _ => walk_expr(self, expr),
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
