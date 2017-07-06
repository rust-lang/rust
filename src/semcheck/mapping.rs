//! The implementation of various map data structures.
//!
//! This module provides facilities to record item correspondence of various kinds, as well as a
//! map used to temporarily match up unsorted item sequences' elements by name.

use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use syntax::ast::Name;

/// A mapping from old to new `DefId`s, as well as exports.
///
/// Exports and simple `DefId` mappings are kept separate to record both kinds of correspondence
/// losslessly. The *access* to the stored data happens through the same API, however.
#[derive(Default)]
pub struct IdMapping {
    /// Toplevel items' old `DefId` mapped to new `DefId`, as well as old and new exports.
    toplevel_mapping: HashMap<DefId, (DefId, Export, Export)>,
    /// Other item's old `DefId` mapped to new `DefId`.
    mapping: HashMap<DefId, DefId>,
    /// Children mapping, allowing us to enumerate descendants in `AdtDef`s.
    child_mapping: HashMap<DefId, BTreeSet<DefId>>,
    /// Set of new defaulted type parameters.
    defaulted_type_params: HashSet<DefId>,
}

impl IdMapping {
    /// Register two exports representing the same item across versions.
    pub fn add_export(&mut self, old: Export, new: Export) -> bool {
        if self.toplevel_mapping.contains_key(&old.def.def_id()) {
            return false;
        }

        self.toplevel_mapping
            .insert(old.def.def_id(), (new.def.def_id(), old, new));

        true
    }

    /// Add any other item pair's old and new `DefId`s.
    pub fn add_item(&mut self, old: DefId, new: DefId) {
        assert!(!self.mapping.contains_key(&old),
                "bug: overwriting {:?} => {:?} with {:?}!",
                old,
                self.mapping[&old],
                new);

        self.mapping.insert(old, new);
    }

    /// Add any other item pair's old and new `DefId`s, together with a parent entry.
    pub fn add_subitem(&mut self, parent: DefId, old: DefId, new: DefId) {
        self.add_item(old, new);
        self.child_mapping
            .entry(parent)
            .or_insert_with(Default::default)
            .insert(old);
    }

    /// Record that a `DefId` represents a newly added defaulted type parameter.
    pub fn add_defaulted_type_param(&mut self, new: DefId) {
        self.defaulted_type_params
            .insert(new);
    }

    /// Check whether a `DefId` represents a newly added defaulted type parameter.
    pub fn is_defaulted_type_param(&self, new: &DefId) -> bool {
        self.defaulted_type_params
            .contains(new)
    }

    /// Get the new `DefId` associated with the given old one.
    pub fn get_new_id(&self, old: DefId) -> DefId {
        if let Some(new) = self.toplevel_mapping.get(&old) {
            new.0
        } else {
            self.mapping[&old]
        }
    }

    /// Tell us whether a `DefId` is present in the mappings.
    pub fn contains_id(&self, old: DefId) -> bool {
        self.toplevel_mapping.contains_key(&old) || self.mapping.contains_key(&old)
    }

    /// Construct a queue of toplevel item pairs' `DefId`s.
    pub fn construct_queue(&self) -> VecDeque<(DefId, DefId)> {
        self.toplevel_mapping
            .values()
            .map(|&(_, old, new)| (old.def.def_id(), new.def.def_id()))
            .collect()
    }

    /// Iterate over the toplevel item pairs.
    pub fn toplevel_values<'a>(&'a self) -> impl Iterator<Item = (Export, Export)> + 'a {
        self.toplevel_mapping
            .values()
            .map(|&(_, old, new)| (old, new))
    }

    /// Iterate over the item pairs of all children of a given item.
    pub fn children_values<'a>(&'a self, parent: DefId)
        -> Option<impl Iterator<Item = (DefId, DefId)> + 'a>
    {
        self.child_mapping
            .get(&parent)
            .map(|m| m.iter().map(move |old| (*old, self.mapping[old])))
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
