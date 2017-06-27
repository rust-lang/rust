use rustc::hir::def::Def;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;

use std::collections::HashMap;

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
        self.toplevel_mapping
            .insert(old.def.def_id(), (new.def.def_id(), old, new))
            .is_some()
    }

    /// Add any other item pair's old and new `DefId`s.
    pub fn add_item(&mut self, old: DefId, new: DefId) {
        self.mapping.insert(old, new);
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
}

/// A representation of a namespace an item belongs to.
#[derive(PartialEq, Eq, Hash)]
pub enum Namespace {
    /// The type namespace.
    Type,
    /// The value namespace.
    Value,
    /// The macro namespace.
    Macro,
    /// No namespace, so to say.
    Err,
}

/// Get an item's namespace.
pub fn get_namespace(def: &Def) -> Namespace {
    use rustc::hir::def::Def::*;

    match *def {
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
        SelfTy(_, _) => Namespace::Type,
        Fn(_) |
        Const(_) |
        Static(_, _) |
        StructCtor(_, _) |
        VariantCtor(_, _) |
        Method(_) |
        AssociatedConst(_) |
        Local(_) |
        Upvar(_, _, _) |
        Label(_) => Namespace::Value,
        Macro(_, _) => Namespace::Macro,
        GlobalAsm(_) |
        Err => Namespace::Err,
    }
}

