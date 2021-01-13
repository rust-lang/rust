use super::{ForeignItem, ImplItem, Item, TraitItem};

/// The "item-like visitor" defines only the top-level methods
/// that can be invoked by `Crate::visit_all_item_likes()`. Whether
/// this trait is the right one to implement will depend on the
/// overall pattern you need. Here are the three available patterns,
/// in roughly the order of desirability:
///
/// 1. **Shallow visit**: Get a simple callback for every item (or item-like thing) in the HIR.
///    - Example: find all items with a `#[foo]` attribute on them.
///    - How: Implement `ItemLikeVisitor` and call `tcx.hir().krate().visit_all_item_likes()`.
///    - Pro: Efficient; just walks the lists of item-like things, not the nodes themselves.
///    - Con: Don't get information about nesting
///    - Con: Don't have methods for specific bits of HIR, like "on
///      every expr, do this".
/// 2. **Deep visit**: Want to scan for specific kinds of HIR nodes within
///    an item, but don't care about how item-like things are nested
///    within one another.
///    - Example: Examine each expression to look for its type and do some check or other.
///    - How: Implement `intravisit::Visitor` and override the `nested_visit_map()` method
///      to return `NestedVisitorMap::OnlyBodies` and use
///      `tcx.hir().krate().visit_all_item_likes(&mut visitor.as_deep_visitor())`. Within
///      your `intravisit::Visitor` impl, implement methods like `visit_expr()` (don't forget
///      to invoke `intravisit::walk_expr()` to keep walking the subparts).
///    - Pro: Visitor methods for any kind of HIR node, not just item-like things.
///    - Pro: Integrates well into dependency tracking.
///    - Con: Don't get information about nesting between items
/// 3. **Nested visit**: Want to visit the whole HIR and you care about the nesting between
///    item-like things.
///    - Example: Lifetime resolution, which wants to bring lifetimes declared on the
///      impl into scope while visiting the impl-items, and then back out again.
///    - How: Implement `intravisit::Visitor` and override the `nested_visit_map()` method
///      to return `NestedVisitorMap::All`. Walk your crate with `intravisit::walk_crate()`
///      invoked on `tcx.hir().krate()`.
///    - Pro: Visitor methods for any kind of HIR node, not just item-like things.
///    - Pro: Preserves nesting information
///    - Con: Does not integrate well into dependency tracking.
///
/// Note: the methods of `ItemLikeVisitor` intentionally have no
/// defaults, so that as we expand the list of item-like things, we
/// revisit the various visitors to see if they need to change. This
/// is harder to do with `intravisit::Visitor`, so when you add a new
/// `visit_nested_foo()` method, it is recommended that you search for
/// existing `fn visit_nested` methods to see where changes are
/// needed.
pub trait ItemLikeVisitor<'hir> {
    fn visit_item(&mut self, item: &'hir Item<'hir>);
    fn visit_trait_item(&mut self, trait_item: &'hir TraitItem<'hir>);
    fn visit_impl_item(&mut self, impl_item: &'hir ImplItem<'hir>);
    fn visit_foreign_item(&mut self, foreign_item: &'hir ForeignItem<'hir>);
}

/// A parallel variant of `ItemLikeVisitor`.
pub trait ParItemLikeVisitor<'hir> {
    fn visit_item(&self, item: &'hir Item<'hir>);
    fn visit_trait_item(&self, trait_item: &'hir TraitItem<'hir>);
    fn visit_impl_item(&self, impl_item: &'hir ImplItem<'hir>);
    fn visit_foreign_item(&self, foreign_item: &'hir ForeignItem<'hir>);
}
