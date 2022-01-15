use rustc_hir::intravisit::nested_filter::NestedFilter;

/// Do not visit nested item-like things, but visit nested things
/// that are inside of an item-like.
///
/// **This is the most common choice.** A very common pattern is
/// to use `visit_all_item_likes()` as an outer loop,
/// and to have the visitor that visits the contents of each item
/// using this setting.
pub struct OnlyBodies(());
impl<'hir> NestedFilter<'hir> for OnlyBodies {
    type Map = crate::hir::map::Map<'hir>;
    const INTER: bool = false;
    const INTRA: bool = true;
}

/// Visits all nested things, including item-likes.
///
/// **This is an unusual choice.** It is used when you want to
/// process everything within their lexical context. Typically you
/// kick off the visit by doing `walk_krate()`.
pub struct All(());
impl<'hir> NestedFilter<'hir> for All {
    type Map = crate::hir::map::Map<'hir>;
    const INTER: bool = true;
    const INTRA: bool = true;
}
