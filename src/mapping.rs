//! The implementation of various map data structures.
//!
//! This module provides facilities to record item correspondence of various kinds, as well as a
//! map used to temporarily match up unsorted item sequences' elements by name.

use rustc_ast::ast::Name;
use rustc_hir::{
    def::Res,
    def_id::{CrateNum, DefId},
    HirId,
};
use rustc_middle::{
    hir::exports::Export,
    ty::{AssocKind, GenericParamDef, GenericParamDefKind},
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// A description of an item found in an inherent impl.
#[derive(Debug, PartialEq)]
pub struct InherentEntry {
    /// The parent item's `DefId`.
    pub parent_def_id: DefId,
    /// The kind of the item.
    pub kind: AssocKind,
    /// The item's name.
    pub name: Name,
}

impl Eq for InherentEntry {}

fn assert_impl_eq<T: Eq>() {}

#[allow(dead_code)]
fn assert_inherent_entry_members_impl_eq() {
    assert_impl_eq::<DefId>();

    // FIXME derive Eq again once AssocKind impls Eq again.
    // assert_impl_eq::<AssocKind>();

    assert_impl_eq::<Name>();
}

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for InherentEntry {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.parent_def_id.hash(hasher);

        // FIXME derive Hash again once AssocKind derives Hash again.
        match self.kind {
            AssocKind::Const => 0u8.hash(hasher),
            AssocKind::Fn => 1u8.hash(hasher),
            AssocKind::OpaqueTy => 2u8.hash(hasher),
            AssocKind::Type => 3u8.hash(hasher),
        }

        self.name.hash(hasher);
    }
}

/// A set of pairs of impl- and item `DefId`s for inherent associated items.
pub type InherentImplSet = BTreeSet<(DefId, DefId)>;

/// A mapping from old to new `DefId`s, as well as associated definitions, if applicable.
///
/// Definitions and simple `DefId` mappings are kept separate to record both kinds of
/// correspondence losslessly. The *access* to the stored data happens through the same API,
/// however. A reverse mapping is also included, but only for `DefId` lookup.
#[cfg_attr(feature = "cargo-clippy", allow(clippy::module_name_repetitions))]
pub struct IdMapping {
    /// The old crate.
    old_crate: CrateNum,
    /// The new crate.
    new_crate: CrateNum,
    /// Toplevel items' old `DefId` mapped to old and new `Res`.
    toplevel_mapping: HashMap<DefId, (Res, Res)>,
    /// The set of items that have been removed or added and thus have no corresponding item in
    /// the other crate.
    non_mapped_items: HashSet<DefId>,
    /// Trait items' old `DefId` mapped to old and new `Res`, and the enclosing trait's `DefId`.
    trait_item_mapping: HashMap<DefId, (Res, Res, DefId)>,
    /// The set of private traits in both crates.
    private_traits: HashSet<DefId>,
    /// Other items' old `DefId` mapped to new `DefId`.
    internal_mapping: HashMap<DefId, DefId>,
    /// Children mapping, allowing us to enumerate descendants in `AdtDef`s.
    child_mapping: HashMap<DefId, BTreeSet<DefId>>,
    /// New `DefId`s mapped to their old counterparts.
    reverse_mapping: HashMap<DefId, DefId>,
    /// Type parameters' `DefId`s mapped to their definitions.
    type_params: HashMap<DefId, GenericParamDef>,
    /// Map from inherent impls' descriptors to the impls they are declared in.
    inherent_items: HashMap<InherentEntry, InherentImplSet>,
}

impl IdMapping {
    /// Construct a new mapping with the given crate information.
    pub fn new(old_crate: CrateNum, new_crate: CrateNum) -> Self {
        Self {
            old_crate,
            new_crate,
            toplevel_mapping: HashMap::new(),
            non_mapped_items: HashSet::new(),
            trait_item_mapping: HashMap::new(),
            private_traits: HashSet::new(),
            internal_mapping: HashMap::new(),
            child_mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
            type_params: HashMap::new(),
            inherent_items: HashMap::new(),
        }
    }

    /// Register two exports representing the same item across versions.
    pub fn add_export(&mut self, old: Res, new: Res) -> bool {
        let (old_def_id, new_def_id) =
            if let (Some(old_def_id), Some(new_def_id)) = (old.opt_def_id(), new.opt_def_id()) {
                (old_def_id, new_def_id)
            } else {
                return false;
            };

        if !self.in_old_crate(old_def_id) || self.toplevel_mapping.contains_key(&old_def_id) {
            return false;
        }

        self.toplevel_mapping.insert(old_def_id, (old, new));
        self.reverse_mapping.insert(new_def_id, old_def_id);

        true
    }

    /// Register that an old item has no corresponding new item.
    pub fn add_non_mapped(&mut self, def_id: DefId) {
        self.non_mapped_items.insert(def_id);
    }

    /// Add any trait item's old and new `DefId`s.
    pub fn add_trait_item(&mut self, old: Res, new: Res, old_trait: DefId) {
        let old_def_id = old.def_id();

        assert!(self.in_old_crate(old_def_id));

        self.trait_item_mapping
            .insert(old_def_id, (old, new, old_trait));
        self.reverse_mapping.insert(new.def_id(), old_def_id);
    }

    /// Add a private trait's `DefId`.
    pub fn add_private_trait(&mut self, trait_def_id: DefId) {
        self.private_traits.insert(trait_def_id);
    }

    /// Add any other item's old and new `DefId`s.
    pub fn add_internal_item(&mut self, old: DefId, new: DefId) {
        assert!(
            !self.internal_mapping.contains_key(&old),
            "bug: overwriting {:?} => {:?} with {:?}!",
            old,
            self.internal_mapping[&old],
            new
        );
        assert!(self.in_old_crate(old));
        assert!(self.in_new_crate(new));

        self.internal_mapping.insert(old, new);
        self.reverse_mapping.insert(new, old);
    }

    /// Add any other item's old and new `DefId`s, together with a parent entry.
    pub fn add_subitem(&mut self, old_parent: DefId, old: DefId, new: DefId) {
        // NB: we rely on the asserts in `add_internal_item` here.
        self.add_internal_item(old, new);
        self.child_mapping
            .entry(old_parent)
            .or_insert_with(Default::default)
            .insert(old);
    }

    /// Record that a `DefId` represents a type parameter.
    pub fn add_type_param(&mut self, param: &GenericParamDef) {
        match param.kind {
            GenericParamDefKind::Lifetime => unreachable!(),
            GenericParamDefKind::Type { .. } => (),
            GenericParamDefKind::Const => unreachable!(),
        };

        self.type_params.insert(param.def_id, param.clone());
    }

    /// Get the type parameter represented by a given `DefId`.
    pub fn get_type_param(&self, did: &DefId) -> &GenericParamDef {
        &self.type_params[did]
    }

    /// Check whether a `DefId` represents a non-mapped defaulted type parameter.
    pub fn is_non_mapped_defaulted_type_param(&self, def_id: DefId) -> bool {
        self.non_mapped_items.contains(&def_id)
            && self
                .type_params
                .get(&def_id)
                .map_or(false, |def| match def.kind {
                    GenericParamDefKind::Type { has_default, .. } => has_default,
                    _ => unreachable!(),
                })
    }

    /// Record an item from an inherent impl.
    pub fn add_inherent_item(
        &mut self,
        parent_def_id: DefId,
        kind: AssocKind,
        name: Name,
        impl_def_id: DefId,
        item_def_id: DefId,
    ) {
        self.inherent_items
            .entry(InherentEntry {
                parent_def_id,
                kind,
                name,
            })
            .or_insert_with(Default::default)
            .insert((impl_def_id, item_def_id));
    }

    /// Get the impl data for an inherent item.
    pub fn get_inherent_impls(&self, inherent_entry: &InherentEntry) -> Option<&InherentImplSet> {
        self.inherent_items.get(inherent_entry)
    }

    /// Get the new `DefId` associated with the given old one.
    pub fn get_new_id(&self, old: DefId) -> Option<DefId> {
        assert!(!self.in_new_crate(old));

        if self.in_old_crate(old) {
            if let Some(new) = self.toplevel_mapping.get(&old) {
                Some(new.1.def_id())
            } else if let Some(new) = self.trait_item_mapping.get(&old) {
                Some(new.1.def_id())
            } else if let Some(new_def_id) = self.internal_mapping.get(&old) {
                Some(*new_def_id)
            } else {
                None
            }
        } else {
            Some(old)
        }
    }

    /// Get the old `DefId` associated with the given new one.
    pub fn get_old_id(&self, new: DefId) -> Option<DefId> {
        assert!(!self.in_old_crate(new));

        if self.in_new_crate(new) {
            self.reverse_mapping.get(&new).cloned()
        } else {
            Some(new)
        }
    }

    /// Return the `DefId` of the trait a given item belongs to.
    pub fn get_trait_def(&self, item_def_id: DefId) -> Option<DefId> {
        self.trait_item_mapping.get(&item_def_id).map(|t| t.2)
    }

    /// Check whether the given `DefId` is a private trait.
    pub fn is_private_trait(&self, trait_def_id: DefId) -> bool {
        self.private_traits.contains(&trait_def_id)
    }

    /// Check whether an old `DefId` is present in the mappings.
    pub fn contains_old_id(&self, old: DefId) -> bool {
        self.toplevel_mapping.contains_key(&old)
            || self.trait_item_mapping.contains_key(&old)
            || self.internal_mapping.contains_key(&old)
    }

    /// Check whether a new `DefId` is present in the mappings.
    pub fn contains_new_id(&self, new: DefId) -> bool {
        self.reverse_mapping.contains_key(&new)
    }

    /// Construct a queue of toplevel item pairs' `DefId`s.
    pub fn toplevel_queue(&self) -> VecDeque<(Res, Res)> {
        self.toplevel_mapping.values().copied().collect()
    }

    /// Iterate over the toplevel and trait item pairs.
    pub fn items<'a>(&'a self) -> impl Iterator<Item = (Res, Res)> + 'a {
        self.toplevel_mapping
            .values()
            .cloned()
            .chain(self.trait_item_mapping.values().map(|&(o, n, _)| (o, n)))
    }

    /// Iterate over the item pairs of all children of a given item.
    pub fn children_of<'a>(
        &'a self,
        parent: DefId,
    ) -> Option<impl Iterator<Item = (DefId, DefId)> + 'a> {
        self.child_mapping
            .get(&parent)
            .map(|m| m.iter().map(move |old| (*old, self.internal_mapping[old])))
    }

    /// Iterate over all items in inherent impls.
    pub fn inherent_impls(&self) -> impl Iterator<Item = (&InherentEntry, &InherentImplSet)> {
        self.inherent_items.iter()
    }

    /// Check whether a `DefId` belongs to an item in the old crate.
    pub fn in_old_crate(&self, did: DefId) -> bool {
        self.old_crate == did.krate
    }

    /// Get the old crate's `CrateNum`.
    pub fn get_old_crate(&self) -> CrateNum {
        self.old_crate
    }

    /// Check whether a `DefId` belongs to an item in the new crate.
    pub fn in_new_crate(&self, did: DefId) -> bool {
        self.new_crate == did.krate
    }

    /// Get the new crate's `CrateNum`.
    pub fn get_new_crate(&self) -> CrateNum {
        self.new_crate
    }
}

/// An export that could be missing from one of the crate versions.
type OptionalExport = Option<Export<HirId>>;

/// A mapping from names to pairs of old and new exports.
///
/// Both old and new exports can be missing. Allows for reuse of the `HashMap`s used for storage.
#[derive(Default)]
#[cfg_attr(feature = "cargo-clippy", allow(clippy::module_name_repetitions))]
pub struct NameMapping {
    /// The exports in the type namespace.
    type_map: HashMap<Name, (OptionalExport, OptionalExport)>,
    /// The exports in the value namespace.
    value_map: HashMap<Name, (OptionalExport, OptionalExport)>,
    /// The exports in the macro namespace.
    macro_map: HashMap<Name, (OptionalExport, OptionalExport)>,
}

impl NameMapping {
    /// Insert a single export in the appropriate map, at the appropriate position.
    fn insert(&mut self, item: Export<HirId>, old: bool) {
        use rustc_hir::def::DefKind::*;
        use rustc_hir::def::Res::*;

        let map = match item.res {
            Def(kind, _) => match kind {
                Mod |
                Struct |
                Union |
                Enum |
                Variant |
                Trait |
                TyAlias |
                ForeignTy |
                TraitAlias | // TODO: will need some handling later on
                AssocTy |
                TyParam |
                OpaqueTy |
                AssocOpaqueTy => Some(&mut self.type_map),
                Fn |
                Const |
                ConstParam |
                Static |
                Ctor(_, _) |
                AssocFn |
                AssocConst => Some(&mut self.value_map),
                Macro(_) => Some(&mut self.macro_map),
            },
            PrimTy(_) | SelfTy(_, _) => Some(&mut self.type_map),
            SelfCtor(_) | Local(_) => Some(&mut self.value_map),
            _ => None,
        };

        if let Some(map) = map {
            if old {
                map.entry(item.ident.name).or_insert((None, None)).0 = Some(item);
            } else {
                map.entry(item.ident.name).or_insert((None, None)).1 = Some(item);
            };
        }
    }

    /// Add all items from two vectors of old/new exports.
    pub fn add(&mut self, old_items: Vec<Export<HirId>>, new_items: Vec<Export<HirId>>) {
        for item in old_items {
            self.insert(item, true);
        }

        for item in new_items {
            self.insert(item, false);
        }
    }

    /// Drain the item pairs being stored.
    pub fn drain<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Option<Export<HirId>>, Option<Export<HirId>>)> + 'a {
        self.type_map
            .drain()
            .chain(self.value_map.drain())
            .chain(self.macro_map.drain())
            .map(|t| t.1)
    }
}
