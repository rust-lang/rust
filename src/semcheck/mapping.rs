//! The implementation of various map data structures.
//!
//! This module provides facilities to record item correspondence of various kinds, as well as a
//! map used to temporarily match up unsorted item sequences' elements by name.

use rustc::hir::def::{Def, Export};
use rustc::hir::def_id::{CrateNum, DefId};
use rustc::ty::TypeParameterDef;

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use syntax::ast::Name;

/// A mapping from old to new `DefId`s, as well as associated definitions, if applicable.
///
/// Definitions and simple `DefId` mappings are kept separate to record both kinds of
/// correspondence losslessly. The *access* to the stored data happens through the same API,
/// however.
pub struct IdMapping {
    /// The old crate.
    old_crate: CrateNum,
    /// The new crate.
    new_crate: CrateNum,
    /// Toplevel items' old `DefId` mapped to old and new `Def`.
    toplevel_mapping: HashMap<DefId, (Def, Def)>,
    /// The set of toplevel items that have been removed.
    removed_items: HashSet<DefId>,
    /// Trait items' old `DefId` mapped to old and new `Def`.
    trait_item_mapping: HashMap<DefId, (Def, Def, DefId)>,
    /// Other items' old `DefId` mapped to new `DefId`.
    internal_mapping: HashMap<DefId, DefId>,
    /// Children mapping, allowing us to enumerate descendants in `AdtDef`s.
    child_mapping: HashMap<DefId, BTreeSet<DefId>>,
    /// Map of type parameters' `DefId`s and their definitions.
    type_params: HashMap<DefId, TypeParameterDef>,
}

impl IdMapping {
    pub fn new(old_crate: CrateNum, new_crate: CrateNum) -> IdMapping {
        IdMapping {
            old_crate: old_crate,
            new_crate: new_crate,
            toplevel_mapping: HashMap::new(),
            removed_items: HashSet::new(),
            trait_item_mapping: HashMap::new(),
            internal_mapping: HashMap::new(),
            child_mapping: HashMap::new(),
            type_params: HashMap::new(),
        }
    }

    /// Register two exports representing the same item across versions.
    pub fn add_export(&mut self, old: Def, new: Def) -> bool {
        let old_def_id = old.def_id();

        if !self.in_old_crate(old_def_id) || self.toplevel_mapping.contains_key(&old_def_id) {
            return false;
        }

        self.toplevel_mapping
            .insert(old_def_id, (old, new));

        true
    }

    /// Register that an old item has no corresponding new item.
    pub fn add_removal(&mut self, old: DefId) {
        self.removed_items.insert(old);
    }

    /// Add any trait item's old and new `DefId`s.
    pub fn add_trait_item(&mut self, old: Def, new: Def, trait_def_id: DefId) {
        let old_def_id = old.def_id();

        assert!(self.in_old_crate(old_def_id));

        self.trait_item_mapping.insert(old_def_id, (old, new, trait_def_id));
    }

    /// Add any other item's old and new `DefId`s.
    pub fn add_internal_item(&mut self, old: DefId, new: DefId) {
        assert!(!self.internal_mapping.contains_key(&old),
                "bug: overwriting {:?} => {:?} with {:?}!",
                old,
                self.internal_mapping[&old],
                new);
        assert!(self.in_old_crate(old));

        self.internal_mapping.insert(old, new);
    }

    /// Add any other item's old and new `DefId`s, together with a parent entry.
    pub fn add_subitem(&mut self, parent: DefId, old: DefId, new: DefId) {
        // NB: we rely on the assers in `add_internal_item` here.
        self.add_internal_item(old, new);
        self.child_mapping
            .entry(parent)
            .or_insert_with(Default::default)
            .insert(old);
    }

    /// Record that a `DefId` represents a new type parameter.
    pub fn add_type_param(&mut self, new: TypeParameterDef) {
        assert!(self.in_new_crate(new.def_id));
        self.type_params.insert(new.def_id, new);
    }

    /// Get the type parameter represented by a given `DefId`.
    pub fn get_type_param(&self, did: &DefId) -> TypeParameterDef {
        self.type_params[did]
    }

    /// Check whether a `DefId` represents a newly added defaulted type parameter.
    pub fn is_defaulted_type_param(&self, new: &DefId) -> bool {
        // TODO?
        self.type_params
            .get(new)
            .map_or(false, |def| def.has_default)
    }

    /// Get the new `DefId` associated with the given old one.
    pub fn get_new_id(&self, old: DefId) -> DefId {
        assert!(!self.in_new_crate(old));

        if self.in_old_crate(old) && !self.removed_items.contains(&old) {
            if let Some(new) = self.toplevel_mapping.get(&old) {
                new.1.def_id()
            } else if let Some(new) = self.trait_item_mapping.get(&old) {
                new.1.def_id()
            } else {
                self.internal_mapping[&old]
            }
        } else {
            old
        }
    }

    /// Get the new `DefId` associated with the given old one, respecting possibly removed
    /// traits that are a parent of the given `DefId`.
    pub fn get_new_trait_item_id(&self, old: DefId, trait_id: DefId) -> DefId {
        assert!(!self.in_new_crate(trait_id));

        if !self.removed_items.contains(&trait_id) {
            self.get_new_id(old)
        } else {
            old
        }
    }

    /// Return the `DefId` of the trait a given item belongs to.
    pub fn get_trait_def(&self, item_def_id: &DefId) -> Option<DefId> {
        self.trait_item_mapping.get(item_def_id).map(|t| t.2)
    }

    /// Check whether a `DefId` is present in the mappings.
    pub fn contains_id(&self, old: DefId) -> bool {
        self.toplevel_mapping.contains_key(&old) ||
            self.trait_item_mapping.contains_key(&old) ||
            self.internal_mapping.contains_key(&old)
    }

    /// Construct a queue of toplevel item pairs' `DefId`s.
    pub fn toplevel_queue(&self) -> VecDeque<(DefId, DefId)> {
        self.toplevel_mapping
            .values()
            .map(|&(old, new)| (old.def_id(), new.def_id()))
            .collect()
    }

    /// Iterate over the toplevel and trait item pairs.
    pub fn items<'a>(&'a self) -> impl Iterator<Item = (Def, Def)> + 'a {
        self.toplevel_mapping
            .values()
            .cloned()
            .chain(self.trait_item_mapping.values().map(|&(o, n, _)| (o, n)))
    }

    /// Iterate over the item pairs of all children of a given item.
    pub fn children_of<'a>(&'a self, parent: DefId)
        -> Option<impl Iterator<Item = (DefId, DefId)> + 'a>
    {
        self.child_mapping
            .get(&parent)
            .map(|m| m.iter().map(move |old| (*old, self.internal_mapping[old])))
    }

    /// Check whether a `DefId` belongs to an item in the old crate.
    pub fn in_old_crate(&self, did: DefId) -> bool {
        self.old_crate == did.krate
    }

    /// Check whether a `DefId` belongs to an item in the new crate.
    pub fn in_new_crate(&self, did: DefId) -> bool {
        self.new_crate == did.krate
    }
}

/// A mapping from names to pairs of old and new exports.
///
/// Both old and new exports can be missing. Allows for reuse of the `HashMap`s used.
#[derive(Default)]
pub struct NameMapping {
    /// The exports in the type namespace.
    type_map: HashMap<Name, (Option<Export>, Option<Export>)>,
    /// The exports in the value namespace.
    value_map: HashMap<Name, (Option<Export>, Option<Export>)>,
    /// The exports in the macro namespace.
    macro_map: HashMap<Name, (Option<Export>, Option<Export>)>,
}

impl NameMapping {
    /// Insert a single export in the appropriate map, at the appropriate position.
    fn insert(&mut self, item: Export, old: bool) {
        use rustc::hir::def::Def::*;

        let map = match item.def {
            Mod(_) |
            Struct(_) |
            Union(_) |
            Enum(_) |
            Variant(_) |
            Trait(_) |
            TyAlias(_) |
            AssociatedTy(_) |
            PrimTy(_) |
            TyParam(_) |
            SelfTy(_, _) => Some(&mut self.type_map),
            Fn(_) |
            Const(_) |
            Static(_, _) |
            StructCtor(_, _) |
            VariantCtor(_, _) |
            Method(_) |
            AssociatedConst(_) |
            Local(_) |
            Upvar(_, _, _) |
            Label(_) => Some(&mut self.value_map),
            Macro(_, _) => Some(&mut self.macro_map),
            GlobalAsm(_) |
            Err => None,
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
    pub fn add(&mut self, mut old_items: Vec<Export>, mut new_items: Vec<Export>) {
        for item in old_items.drain(..) {
            self.insert(item, true);
        }

        for item in new_items.drain(..) {
            self.insert(item, false);
        }
    }

    /// Drain the item pairs being stored.
    pub fn drain<'a>(&'a mut self)
        -> impl Iterator<Item = (Option<Export>, Option<Export>)> + 'a
    {
        self.type_map
            .drain()
            .chain(self.value_map.drain())
            .chain(self.macro_map.drain())
            .map(|t| t.1)
    }
}
