use rustc_data_structures::sorted_map::SortedIndexMultiMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_span::symbol::{Ident, Symbol};

use super::{TyCtxt, Visibility};
use crate::ty;

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable, Hash, Encodable, Decodable)]
pub enum AssocItemContainer {
    Trait,
    Impl,
}

/// Information about an associated item
#[derive(Copy, Clone, Debug, PartialEq, HashStable, Eq, Hash, Encodable, Decodable)]
pub struct AssocItem {
    pub def_id: DefId,
    pub name: Symbol,
    pub kind: AssocKind,
    pub container: AssocItemContainer,

    /// If this is an item in an impl of a trait then this is the `DefId` of
    /// the associated item on the trait that this implements.
    pub trait_item_def_id: Option<DefId>,

    /// Whether this is a method with an explicit self
    /// as its first parameter, allowing method calls.
    pub fn_has_self_parameter: bool,

    /// `Some` if the associated item (an associated type) comes from the
    /// return-position `impl Trait` in trait desugaring. The `ImplTraitInTraitData`
    /// provides additional information about its source.
    pub opt_rpitit_info: Option<ty::ImplTraitInTraitData>,
}

impl AssocItem {
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.def_id).unwrap())
    }

    /// Gets the defaultness of the associated item.
    /// To get the default associated type, use the [`type_of`] query on the
    /// [`DefId`] of the type.
    ///
    /// [`type_of`]: crate::ty::TyCtxt::type_of
    pub fn defaultness(&self, tcx: TyCtxt<'_>) -> hir::Defaultness {
        tcx.defaultness(self.def_id)
    }

    #[inline]
    pub fn visibility(&self, tcx: TyCtxt<'_>) -> Visibility<DefId> {
        tcx.visibility(self.def_id)
    }

    #[inline]
    pub fn container_id(&self, tcx: TyCtxt<'_>) -> DefId {
        tcx.parent(self.def_id)
    }

    #[inline]
    pub fn trait_container(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.container {
            AssocItemContainer::Impl => None,
            AssocItemContainer::Trait => Some(tcx.parent(self.def_id)),
        }
    }

    #[inline]
    pub fn impl_container(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.container {
            AssocItemContainer::Impl => Some(tcx.parent(self.def_id)),
            AssocItemContainer::Trait => None,
        }
    }

    pub fn signature(&self, tcx: TyCtxt<'_>) -> String {
        match self.kind {
            ty::AssocKind::Fn => {
                // We skip the binder here because the binder would deanonymize all
                // late-bound regions, and we don't want method signatures to show up
                // `as for<'r> fn(&'r MyType)`. Pretty-printing handles late-bound
                // regions just fine, showing `fn(&MyType)`.
                tcx.fn_sig(self.def_id).instantiate_identity().skip_binder().to_string()
            }
            ty::AssocKind::Type => format!("type {};", self.name),
            ty::AssocKind::Const => {
                format!(
                    "const {}: {:?};",
                    self.name,
                    tcx.type_of(self.def_id).instantiate_identity()
                )
            }
        }
    }

    pub fn descr(&self) -> &'static str {
        match self.kind {
            ty::AssocKind::Const => "const",
            ty::AssocKind::Fn if self.fn_has_self_parameter => "method",
            ty::AssocKind::Fn => "associated function",
            ty::AssocKind::Type => "type",
        }
    }

    pub fn is_impl_trait_in_trait(&self) -> bool {
        self.opt_rpitit_info.is_some()
    }
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable, Eq, Hash, Encodable, Decodable)]
pub enum AssocKind {
    Const,
    Fn,
    Type,
}

impl AssocKind {
    pub fn namespace(&self) -> Namespace {
        match *self {
            ty::AssocKind::Type => Namespace::TypeNS,
            ty::AssocKind::Const | ty::AssocKind::Fn => Namespace::ValueNS,
        }
    }

    pub fn as_def_kind(&self) -> DefKind {
        match self {
            AssocKind::Const => DefKind::AssocConst,
            AssocKind::Fn => DefKind::AssocFn,
            AssocKind::Type => DefKind::AssocTy,
        }
    }
}

impl std::fmt::Display for AssocKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssocKind::Fn => write!(f, "method"),
            AssocKind::Const => write!(f, "associated const"),
            AssocKind::Type => write!(f, "associated type"),
        }
    }
}

/// A list of `ty::AssocItem`s in definition order that allows for efficient lookup by name.
///
/// When doing lookup by name, we try to postpone hygienic comparison for as long as possible since
/// it is relatively expensive. Instead, items are indexed by `Symbol` and hygienic comparison is
/// done only on items with the same name.
#[derive(Debug, Clone, PartialEq, HashStable)]
pub struct AssocItems {
    items: SortedIndexMultiMap<u32, Symbol, ty::AssocItem>,
}

impl AssocItems {
    /// Constructs an `AssociatedItems` map from a series of `ty::AssocItem`s in definition order.
    pub fn new(items_in_def_order: impl IntoIterator<Item = ty::AssocItem>) -> Self {
        let items = items_in_def_order.into_iter().map(|item| (item.name, item)).collect();
        AssocItems { items }
    }

    /// Returns a slice of associated items in the order they were defined.
    ///
    /// New code should avoid relying on definition order. If you need a particular associated item
    /// for a known trait, make that trait a lang item instead of indexing this array.
    pub fn in_definition_order(&self) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.iter().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns an iterator over all associated items with the given name, ignoring hygiene.
    pub fn filter_by_name_unhygienic(
        &self,
        name: Symbol,
    ) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.get_by_key(name)
    }

    /// Returns the associated item with the given name and `AssocKind`, if one exists.
    pub fn find_by_name_and_kind(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        kind: AssocKind,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.kind == kind)
            .find(|item| tcx.hygienic_eq(ident, item.ident(tcx), parent_def_id))
    }

    /// Returns the associated item with the given name and any of `AssocKind`, if one exists.
    pub fn find_by_name_and_kinds(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        // Sorted in order of what kinds to look at
        kinds: &[AssocKind],
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        kinds.iter().find_map(|kind| self.find_by_name_and_kind(tcx, ident, *kind, parent_def_id))
    }

    /// Returns the associated item with the given name in the given `Namespace`, if one exists.
    pub fn find_by_name_and_namespace(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        ns: Namespace,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.kind.namespace() == ns)
            .find(|item| tcx.hygienic_eq(ident, item.ident(tcx), parent_def_id))
    }
}
