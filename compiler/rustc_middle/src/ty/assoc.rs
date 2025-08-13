use rustc_data_structures::sorted_map::SortedIndexMultiMap;
use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_hir::find_attr;
use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_span::{ErrorGuaranteed, Ident, Symbol};

use super::{TyCtxt, Visibility};
use crate::ty;

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable, Hash, Encodable, Decodable)]
pub enum AssocContainer {
    Trait,
    InherentImpl,
    /// The `DefId` points to the trait item being implemented.
    TraitImpl(Result<DefId, ErrorGuaranteed>),
}

/// Information about an associated item
#[derive(Copy, Clone, Debug, PartialEq, HashStable, Eq, Hash, Encodable, Decodable)]
pub struct AssocItem {
    pub def_id: DefId,
    pub kind: AssocKind,
    pub container: AssocContainer,
}

impl AssocItem {
    // Gets the identifier, if it has one.
    pub fn opt_name(&self) -> Option<Symbol> {
        match self.kind {
            ty::AssocKind::Type { data: AssocTypeData::Normal(name) } => Some(name),
            ty::AssocKind::Type { data: AssocTypeData::Rpitit(_) } => None,
            ty::AssocKind::Const { name } => Some(name),
            ty::AssocKind::Fn { name, .. } => Some(name),
        }
    }

    // Gets the identifier name. Aborts if it lacks one, i.e. is an RPITIT
    // associated type.
    pub fn name(&self) -> Symbol {
        self.opt_name().expect("name of non-Rpitit assoc item")
    }

    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name(), tcx.def_ident_span(self.def_id).unwrap())
    }

    /// Gets the defaultness of the associated item.
    /// To get the default associated type, use the [`type_of`] query on the
    /// [`DefId`] of the type.
    ///
    /// [`type_of`]: crate::ty::TyCtxt::type_of
    pub fn defaultness(&self, tcx: TyCtxt<'_>) -> hir::Defaultness {
        match self.container {
            AssocContainer::InherentImpl => hir::Defaultness::Final,
            AssocContainer::Trait | AssocContainer::TraitImpl(_) => tcx.defaultness(self.def_id),
        }
    }

    pub fn expect_trait_impl(&self) -> Result<DefId, ErrorGuaranteed> {
        let AssocContainer::TraitImpl(trait_item_id) = self.container else {
            bug!("expected item to be in a trait impl: {:?}", self.def_id);
        };
        trait_item_id
    }

    /// If this is a trait impl item, returns the `DefId` of the trait item this implements.
    /// Otherwise, returns `DefId` for self. Returns an Err in case the trait item was not
    /// resolved successfully.
    pub fn trait_item_or_self(&self) -> Result<DefId, ErrorGuaranteed> {
        match self.container {
            AssocContainer::TraitImpl(id) => id,
            AssocContainer::Trait | AssocContainer::InherentImpl => Ok(self.def_id),
        }
    }

    pub fn trait_item_def_id(&self) -> Option<DefId> {
        match self.container {
            AssocContainer::TraitImpl(Ok(id)) => Some(id),
            _ => None,
        }
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
            AssocContainer::InherentImpl | AssocContainer::TraitImpl(_) => None,
            AssocContainer::Trait => Some(tcx.parent(self.def_id)),
        }
    }

    #[inline]
    pub fn impl_container(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.container {
            AssocContainer::InherentImpl | AssocContainer::TraitImpl(_) => {
                Some(tcx.parent(self.def_id))
            }
            AssocContainer::Trait => None,
        }
    }

    pub fn signature(&self, tcx: TyCtxt<'_>) -> String {
        match self.kind {
            ty::AssocKind::Fn { .. } => {
                // We skip the binder here because the binder would deanonymize all
                // late-bound regions, and we don't want method signatures to show up
                // `as for<'r> fn(&'r MyType)`. Pretty-printing handles late-bound
                // regions just fine, showing `fn(&MyType)`.
                tcx.fn_sig(self.def_id).instantiate_identity().skip_binder().to_string()
            }
            ty::AssocKind::Type { .. } => format!("type {};", self.name()),
            ty::AssocKind::Const { name } => {
                format!("const {}: {:?};", name, tcx.type_of(self.def_id).instantiate_identity())
            }
        }
    }

    pub fn descr(&self) -> &'static str {
        match self.kind {
            ty::AssocKind::Const { .. } => "associated const",
            ty::AssocKind::Fn { has_self: true, .. } => "method",
            ty::AssocKind::Fn { has_self: false, .. } => "associated function",
            ty::AssocKind::Type { .. } => "associated type",
        }
    }

    pub fn namespace(&self) -> Namespace {
        match self.kind {
            ty::AssocKind::Type { .. } => Namespace::TypeNS,
            ty::AssocKind::Const { .. } | ty::AssocKind::Fn { .. } => Namespace::ValueNS,
        }
    }

    pub fn as_def_kind(&self) -> DefKind {
        match self.kind {
            AssocKind::Const { .. } => DefKind::AssocConst,
            AssocKind::Fn { .. } => DefKind::AssocFn,
            AssocKind::Type { .. } => DefKind::AssocTy,
        }
    }
    pub fn is_type(&self) -> bool {
        matches!(self.kind, ty::AssocKind::Type { .. })
    }

    pub fn is_fn(&self) -> bool {
        matches!(self.kind, ty::AssocKind::Fn { .. })
    }

    pub fn is_method(&self) -> bool {
        matches!(self.kind, ty::AssocKind::Fn { has_self: true, .. })
    }

    pub fn as_tag(&self) -> AssocTag {
        match self.kind {
            AssocKind::Const { .. } => AssocTag::Const,
            AssocKind::Fn { .. } => AssocTag::Fn,
            AssocKind::Type { .. } => AssocTag::Type,
        }
    }

    pub fn is_impl_trait_in_trait(&self) -> bool {
        matches!(self.kind, AssocKind::Type { data: AssocTypeData::Rpitit(_) })
    }

    /// Returns true if:
    /// - This trait associated item has the `#[type_const]` attribute,
    /// - If it is in a trait impl, the item from the original trait has this attribute, or
    /// - It is an inherent assoc const.
    pub fn is_type_const_capable(&self, tcx: TyCtxt<'_>) -> bool {
        if !matches!(self.kind, ty::AssocKind::Const { .. }) {
            return false;
        }

        let def_id = match self.container {
            AssocContainer::Trait => self.def_id,
            AssocContainer::TraitImpl(Ok(trait_item_did)) => trait_item_did,
            AssocContainer::TraitImpl(Err(_)) => return false,
            AssocContainer::InherentImpl => return true,
        };
        find_attr!(tcx.get_all_attrs(def_id), AttributeKind::TypeConst(_))
    }
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable, Eq, Hash, Encodable, Decodable)]
pub enum AssocTypeData {
    Normal(Symbol),
    /// The associated type comes from an RPITIT. It has no name, and the
    /// `ImplTraitInTraitData` provides additional information about its
    /// source.
    Rpitit(ty::ImplTraitInTraitData),
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable, Eq, Hash, Encodable, Decodable)]
pub enum AssocKind {
    Const { name: Symbol },
    Fn { name: Symbol, has_self: bool },
    Type { data: AssocTypeData },
}

impl AssocKind {
    pub fn namespace(&self) -> Namespace {
        match *self {
            ty::AssocKind::Type { .. } => Namespace::TypeNS,
            ty::AssocKind::Const { .. } | ty::AssocKind::Fn { .. } => Namespace::ValueNS,
        }
    }

    pub fn as_def_kind(&self) -> DefKind {
        match self {
            AssocKind::Const { .. } => DefKind::AssocConst,
            AssocKind::Fn { .. } => DefKind::AssocFn,
            AssocKind::Type { .. } => DefKind::AssocTy,
        }
    }
}

impl std::fmt::Display for AssocKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssocKind::Fn { has_self: true, .. } => write!(f, "method"),
            AssocKind::Fn { has_self: false, .. } => write!(f, "associated function"),
            AssocKind::Const { .. } => write!(f, "associated const"),
            AssocKind::Type { .. } => write!(f, "associated type"),
        }
    }
}

// Like `AssocKind`, but just the tag, no fields. Used in various kinds of matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssocTag {
    Const,
    Fn,
    Type,
}

/// A list of `ty::AssocItem`s in definition order that allows for efficient lookup by name.
///
/// When doing lookup by name, we try to postpone hygienic comparison for as long as possible since
/// it is relatively expensive. Instead, items are indexed by `Symbol` and hygienic comparison is
/// done only on items with the same name.
#[derive(Debug, Clone, PartialEq, HashStable)]
pub struct AssocItems {
    items: SortedIndexMultiMap<u32, Option<Symbol>, ty::AssocItem>,
}

impl AssocItems {
    /// Constructs an `AssociatedItems` map from a series of `ty::AssocItem`s in definition order.
    pub fn new(items_in_def_order: impl IntoIterator<Item = ty::AssocItem>) -> Self {
        let items = items_in_def_order.into_iter().map(|item| (item.opt_name(), item)).collect();
        AssocItems { items }
    }

    /// Returns an iterator over associated items in the order they were defined.
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
    ///
    /// Panics if `name.is_empty()` returns `true`.
    pub fn filter_by_name_unhygienic(
        &self,
        name: Symbol,
    ) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        assert!(!name.is_empty());
        self.items.get_by_key(Some(name))
    }

    /// Returns the associated item with the given identifier and `AssocKind`, if one exists.
    /// The identifier is ignoring hygiene. This is meant to be used for lints and diagnostics.
    pub fn filter_by_name_unhygienic_and_kind(
        &self,
        name: Symbol,
        assoc_tag: AssocTag,
    ) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.filter_by_name_unhygienic(name).filter(move |item| item.as_tag() == assoc_tag)
    }

    /// Returns the associated item with the given identifier and `AssocKind`, if one exists.
    /// The identifier is matched hygienically.
    pub fn find_by_ident_and_kind(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        assoc_tag: AssocTag,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.as_tag() == assoc_tag)
            .find(|item| tcx.hygienic_eq(ident, item.ident(tcx), parent_def_id))
    }

    /// Returns the associated item with the given identifier in the given `Namespace`, if one
    /// exists. The identifier is matched hygienically.
    pub fn find_by_ident_and_namespace(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        ns: Namespace,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.namespace() == ns)
            .find(|item| tcx.hygienic_eq(ident, item.ident(tcx), parent_def_id))
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Given an `fn_def_id` of a trait or a trait implementation:
    ///
    /// if `fn_def_id` is a function defined inside a trait, then it synthesizes
    /// a new def id corresponding to a new associated type for each return-
    /// position `impl Trait` in the signature.
    ///
    /// if `fn_def_id` is a function inside of an impl, then for each synthetic
    /// associated type generated for the corresponding trait function described
    /// above, synthesize a corresponding associated type in the impl.
    pub fn associated_types_for_impl_traits_in_associated_fn(
        self,
        fn_def_id: DefId,
    ) -> &'tcx [DefId] {
        let parent_def_id = self.parent(fn_def_id);
        &self.associated_types_for_impl_traits_in_trait_or_impl(parent_def_id)[&fn_def_id]
    }
}
