use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

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
