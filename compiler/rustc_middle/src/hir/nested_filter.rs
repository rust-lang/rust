use rustc_hir::intravisit::nested_filter::NestedFilter;

use crate::ty::TyCtxt;

/// Do not visit nested item-like things, but visit nested things
/// that are inside of an item-like.
///
/// Notably, possible occurrences of bodies in non-item-like things
/// include: closures/coroutines, inline `const {}` blocks, and
/// constant arguments of types, e.g. in `let _: [(); /* HERE */];`.
///
/// **This is the most common choice.** A very common pattern is
/// to use `visit_all_item_likes_in_crate()` as an outer loop,
/// and to have the visitor that visits the contents of each item
/// using this setting.
pub struct OnlyBodies(());
impl<'tcx> NestedFilter<'tcx> for OnlyBodies {
    type MaybeTyCtxt = TyCtxt<'tcx>;
    const INTER: bool = false;
    const INTRA: bool = true;
}

/// Visits all nested things, including item-likes.
///
/// **This is an unusual choice.** It is used when you want to
/// process everything within their lexical context. Typically you
/// kick off the visit by doing `walk_krate()`.
pub struct All(());
impl<'tcx> NestedFilter<'tcx> for All {
    type MaybeTyCtxt = TyCtxt<'tcx>;
    const INTER: bool = true;
    const INTRA: bool = true;
}
