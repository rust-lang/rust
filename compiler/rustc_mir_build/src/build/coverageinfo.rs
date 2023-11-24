use rustc_middle::hir;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::{ExpnKind, Span};

/// If the given item is eligible for coverage instrumentation, collect relevant
/// HIR information that will be needed by the instrumentor pass.
pub(crate) fn make_coverage_hir_info_if_eligible(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<Box<mir::coverage::HirInfo>> {
    assert!(tcx.sess.instrument_coverage());

    is_eligible_for_coverage(tcx, def_id).then(|| Box::new(make_coverage_hir_info(tcx, def_id)))
}

fn is_eligible_for_coverage(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let is_fn_like = tcx.hir().get_by_def_id(def_id).fn_kind().is_some();

    // Only instrument functions, methods, and closures (not constants since they are evaluated
    // at compile time by Miri).
    // FIXME(#73156): Handle source code coverage in const eval, but note, if and when const
    // expressions get coverage spans, we will probably have to "carve out" space for const
    // expressions from coverage spans in enclosing MIR's, like we do for closures. (That might
    // be tricky if const expressions have no corresponding statements in the enclosing MIR.
    // Closures are carved out by their initial `Assign` statement.)
    if !is_fn_like {
        return false;
    }

    let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_COVERAGE) {
        return false;
    }

    true
}

fn make_coverage_hir_info(tcx: TyCtxt<'_>, def_id: LocalDefId) -> mir::coverage::HirInfo {
    let (maybe_fn_sig, hir_body) = fn_sig_and_body(tcx, def_id);

    let function_source_hash = hash_mir_source(tcx, hir_body);
    let body_span = get_body_span(tcx, hir_body, def_id);

    let spans_are_compatible = {
        let source_map = tcx.sess.source_map();
        |a: Span, b: Span| {
            a.eq_ctxt(b)
                && source_map.lookup_source_file_idx(a.lo())
                    == source_map.lookup_source_file_idx(b.lo())
        }
    };

    let fn_sig_span = if let Some(fn_sig) = maybe_fn_sig
        && spans_are_compatible(fn_sig.span, body_span)
        && fn_sig.span.lo() <= body_span.lo()
    {
        fn_sig.span.with_hi(body_span.lo())
    } else {
        body_span.shrink_to_lo()
    };

    mir::coverage::HirInfo { function_source_hash, fn_sig_span, body_span }
}

fn fn_sig_and_body(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> (Option<&rustc_hir::FnSig<'_>>, &rustc_hir::Body<'_>) {
    // FIXME(#79625): Consider improving MIR to provide the information needed, to avoid going back
    // to HIR for it.
    let hir_node = tcx.hir().get_by_def_id(def_id);
    let (_, fn_body_id) =
        hir::map::associated_body(hir_node).expect("HIR node is a function with body");
    (hir_node.fn_sig(), tcx.hir().body(fn_body_id))
}

fn get_body_span<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_body: &rustc_hir::Body<'tcx>,
    def_id: LocalDefId,
) -> Span {
    let mut body_span = hir_body.value.span;

    if tcx.is_closure(def_id.to_def_id()) {
        // If the MIR function is a closure, and if the closure body span
        // starts from a macro, but it's content is not in that macro, try
        // to find a non-macro callsite, and instrument the spans there
        // instead.
        loop {
            let expn_data = body_span.ctxt().outer_expn_data();
            if expn_data.is_root() {
                break;
            }
            if let ExpnKind::Macro { .. } = expn_data.kind {
                body_span = expn_data.call_site;
            } else {
                break;
            }
        }
    }

    body_span
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx rustc_hir::Body<'tcx>) -> u64 {
    // FIXME(cjgillot) Stop hashing HIR manually here.
    let owner = hir_body.id().hir_id.owner;
    tcx.hir_owner_nodes(owner)
        .unwrap()
        .opt_hash_including_bodies
        .unwrap()
        .to_smaller_hash()
        .as_u64()
}
