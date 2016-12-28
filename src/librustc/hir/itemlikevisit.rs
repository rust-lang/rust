// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{Item, ImplItem, TraitItem};
use super::intravisit::Visitor;

/// The "item-like visitor" visitor defines only the top-level methods
/// that can be invoked by `Crate::visit_all_item_likes()`. Whether
/// this trait is the right one to implement will depend on the
/// overall pattern you need. Here are the three available patterns,
/// in roughly the order of desirability:
///
/// 1. **Shallow visit**: Get a simple callback for every item (or item-like thing) in the HIR.
///    - Example: find all items with a `#[foo]` attribute on them.
///    - How: Implement `ItemLikeVisitor` and call `tcx.visit_all_item_likes_in_krate()`.
///    - Pro: Efficient; just walks the lists of item-like things, not the nodes themselves.
///    - Pro: Integrates well into dependency tracking.
///    - Con: Don't get information about nesting
///    - Con: Don't have methods for specific bits of HIR, like "on
///      every expr, do this".
/// 2. **Deep visit**: Want to scan for specific kinds of HIR nodes within
///    an item, but don't care about how item-like things are nested
///    within one another.
///    - Example: Examine each expression to look for its type and do some check or other.
///    - How: Implement `intravisit::Visitor` and use
///      `tcx.visit_all_item_likes_in_krate(visitor.as_deep_visitor())`. Within
///      your `intravisit::Visitor` impl, implement methods like
///      `visit_expr()`; don't forget to invoke
///      `intravisit::walk_visit_expr()` to keep walking the subparts.
///    - Pro: Visitor methods for any kind of HIR node, not just item-like things.
///    - Pro: Integrates well into dependency tracking.
///    - Con: Don't get information about nesting between items
/// 3. **Nested visit**: Want to visit the whole HIR and you care about the nesting between
///    item-like things.
///    - Example: Lifetime resolution, which wants to bring lifetimes declared on the
///      impl into scope while visiting the impl-items, and then back out again.
///    - How: Implement `intravisit::Visitor` and override the
///      `visit_nested_map()` methods to return
///      `NestedVisitorMap::All`. Walk your crate with
///      `intravisit::walk_crate()` invoked on `tcx.map.krate()`.
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
    fn visit_item(&mut self, item: &'hir Item);
    fn visit_trait_item(&mut self, trait_item: &'hir TraitItem);
    fn visit_impl_item(&mut self, impl_item: &'hir ImplItem);
}

pub struct DeepVisitor<'v, V: 'v> {
    visitor: &'v mut V,
}

impl<'v, 'hir, V> DeepVisitor<'v, V>
    where V: Visitor<'hir> + 'v
{
    pub fn new(base: &'v mut V) -> Self {
        DeepVisitor { visitor: base }
    }
}

impl<'v, 'hir, V> ItemLikeVisitor<'hir> for DeepVisitor<'v, V>
    where V: Visitor<'hir>
{
    fn visit_item(&mut self, item: &'hir Item) {
        self.visitor.visit_item(item);
    }

    fn visit_trait_item(&mut self, trait_item: &'hir TraitItem) {
        self.visitor.visit_trait_item(trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'hir ImplItem) {
        self.visitor.visit_impl_item(impl_item);
    }
}
