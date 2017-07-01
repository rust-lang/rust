use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;

use std::collections::{HashMap, VecDeque};

use syntax::ast::Name;

/// A mapping from old to new `DefId`s, as well as exports.
///
/// Exports and simple `DefId` mappings are kept separate to record both kinds of correspondence
/// losslessly. The *access* to the stored data happens through the same API, however.
#[derive(Default)]
pub struct IdMapping {
    /// Toplevel items' old `DefId` mapped to new `DefId`, as well as old and new exports.
    pub toplevel_mapping: HashMap<DefId, (DefId, Export, Export)>,
    /// Other item's old `DefId` mapped to new `DefId`.
    mapping: HashMap<DefId, DefId>,
}

impl IdMapping {
    /// Register two exports representing the same item across versions.
    pub fn add_export(&mut self, old: Export, new: Export) -> bool {
        if !self.toplevel_mapping.contains_key(&old.def.def_id()) {
            self.toplevel_mapping
                .insert(old.def.def_id(), (new.def.def_id(), old, new));
            return true;
        }

        false
    }

    /// Add any other item pair's old and new `DefId`s.
    pub fn add_item(&mut self, old: DefId, new: DefId) {
        if !self.mapping.contains_key(&old) {
            self.mapping.insert(old, new);
        } else {
            panic!("bug: overwriting {:?} => {:?} with {:?}!", old, self.mapping[&old], new);
        }
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

    pub fn construct_queue(&self) -> VecDeque<(DefId, DefId)> {
        self.toplevel_mapping
            .values()
            .map(|&(_, old, new)| (old.def.def_id(), new.def.def_id()))
            .collect()
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
