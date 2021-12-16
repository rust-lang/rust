pub use self::AssocItemContainer::*;

use crate::ty;
use rustc_data_structures::sorted_map::SortedIndexMultiMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_span::symbol::{Ident, Symbol};

use super::{TyCtxt, Visibility};

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable, Hash)]
pub enum AssocItemContainer {
    TraitContainer(DefId),
    ImplContainer(DefId),
}

impl AssocItemContainer {
    pub fn impl_def_id(&self) -> Option<DefId> {
        match *self {
            ImplContainer(id) => Some(id),
            _ => None,
        }
    }

    /// Asserts that this is the `DefId` of an associated item declared
    /// in a trait, and returns the trait `DefId`.
    pub fn assert_trait(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            _ => bug!("associated item has wrong container type: {:?}", self),
        }
    }

    pub fn id(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable, Eq, Hash)]
pub struct AssocItem {
    pub def_id: DefId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssocKind,
    pub vis: Visibility,
    pub defaultness: hir::Defaultness,
    pub container: AssocItemContainer,

    /// Whether this is a method with an explicit self
    /// as its first parameter, allowing method calls.
    pub fn_has_self_parameter: bool,
}

impl AssocItem {
    pub fn signature(&self, tcx: TyCtxt<'_>) -> String {
        match self.kind {
            ty::AssocKind::Fn => {
                // We skip the binder here because the binder would deanonymize all
                // late-bound regions, and we don't want method signatures to show up
                // `as for<'r> fn(&'r MyType)`.  Pretty-printing handles late-bound
                // regions just fine, showing `fn(&MyType)`.
                tcx.fn_sig(self.def_id).skip_binder().to_string()
            }
            ty::AssocKind::Type => format!("type {};", self.ident),
            ty::AssocKind::Const => {
                format!("const {}: {:?};", self.ident, tcx.type_of(self.def_id))
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable, Eq, Hash)]
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

/// A list of `ty::AssocItem`s in definition order that allows for efficient lookup by name.
///
/// When doing lookup by name, we try to postpone hygienic comparison for as long as possible since
/// it is relatively expensive. Instead, items are indexed by `Symbol` and hygienic comparison is
/// done only on items with the same name.
#[derive(Debug, Clone, PartialEq, HashStable)]
pub struct AssocItems<'tcx> {
    pub(super) items: SortedIndexMultiMap<u32, Symbol, &'tcx ty::AssocItem>,
}

impl<'tcx> AssocItems<'tcx> {
    /// Constructs an `AssociatedItems` map from a series of `ty::AssocItem`s in definition order.
    pub fn new(items_in_def_order: impl IntoIterator<Item = &'tcx ty::AssocItem>) -> Self {
        let items = items_in_def_order.into_iter().map(|item| (item.ident.name, item)).collect();
        AssocItems { items }
    }

    /// Returns a slice of associated items in the order they were defined.
    ///
    /// New code should avoid relying on definition order. If you need a particular associated item
    /// for a known trait, make that trait a lang item instead of indexing this array.
    pub fn in_definition_order(&self) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.iter().map(|(_, v)| *v)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns an iterator over all associated items with the given name, ignoring hygiene.
    pub fn filter_by_name_unhygienic(
        &self,
        name: Symbol,
    ) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.get_by_key(name).copied()
    }

    /// Returns an iterator over all associated items with the given name.
    ///
    /// Multiple items may have the same name if they are in different `Namespace`s. For example,
    /// an associated type can have the same name as a method. Use one of the `find_by_name_and_*`
    /// methods below if you know which item you are looking for.
    pub fn filter_by_name<'a>(
        &'a self,
        tcx: TyCtxt<'a>,
        ident: Ident,
        parent_def_id: DefId,
    ) -> impl 'a + Iterator<Item = &'a ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(move |item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
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
            .find(|item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
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
            .find(|item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
    }
}
