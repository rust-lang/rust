use rustc_hir as hir;
use rustc_hir::Expr;
use rustc_hir_typeck::{cast, FnCtxt, Inherited};
use rustc_lint::LateContext;
use rustc_middle::ty::{cast::CastKind, Ty};
use rustc_span::DUMMY_SP;

// check if the component types of the transmuted collection and the result have different ABI,
// size or alignment
pub(super) fn is_layout_incompatible<'tcx>(cx: &LateContext<'tcx>, from: Ty<'tcx>, to: Ty<'tcx>) -> bool {
    if let Ok(from) = cx.tcx.try_normalize_erasing_regions(cx.param_env, from)
        && let Ok(to) = cx.tcx.try_normalize_erasing_regions(cx.param_env, to)
        && let Ok(from_layout) = cx.tcx.layout_of(cx.param_env.and(from))
        && let Ok(to_layout) = cx.tcx.layout_of(cx.param_env.and(to))
    {
        from_layout.size != to_layout.size || from_layout.align.abi != to_layout.align.abi
    } else {
        // no idea about layout, so don't lint
        false
    }
}

/// If a cast from `from_ty` to `to_ty` is valid, returns an Ok containing the kind of
/// the cast. In certain cases, including some invalid casts from array references
/// to pointers, this may cause additional errors to be emitted and/or ICE error
/// messages. This function will panic if that occurs.
pub(super) fn check_cast<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
) -> Option<CastKind> {
    let hir_id = e.hir_id;
    let local_def_id = hir_id.owner.def_id;

    let inherited = Inherited::new(cx.tcx, local_def_id);
    let fn_ctxt = FnCtxt::new(&inherited, cx.param_env, local_def_id);

    // If we already have errors, we can't be sure we can pointer cast.
    assert!(
        !fn_ctxt.errors_reported_since_creation(),
        "Newly created FnCtxt contained errors"
    );

    if let Ok(check) = cast::CastCheck::new(
        &fn_ctxt,
        e,
        from_ty,
        to_ty,
        // We won't show any error to the user, so we don't care what the span is here.
        DUMMY_SP,
        DUMMY_SP,
        hir::Constness::NotConst,
    ) {
        let res = check.do_check(&fn_ctxt);

        // do_check's documentation says that it might return Ok and create
        // errors in the fcx instead of returning Err in some cases. Those cases
        // should be filtered out before getting here.
        assert!(
            !fn_ctxt.errors_reported_since_creation(),
            "`fn_ctxt` contained errors after cast check!"
        );

        res.ok()
    } else {
        None
    }
}
