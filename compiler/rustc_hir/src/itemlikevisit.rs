use super::{ForeignItem, ImplItem, Item, TraitItem};

/// A parallel variant of `ItemLikeVisitor`.
pub trait ParItemLikeVisitor<'hir> {
    fn visit_item(&self, item: &'hir Item<'hir>);
    fn visit_trait_item(&self, trait_item: &'hir TraitItem<'hir>);
    fn visit_impl_item(&self, impl_item: &'hir ImplItem<'hir>);
    fn visit_foreign_item(&self, foreign_item: &'hir ForeignItem<'hir>);
}
