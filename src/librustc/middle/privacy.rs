// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.

pub use self::PrivateDep::*;
pub use self::ImportUse::*;
pub use self::LastPrivate::*;

use middle::def_id::DefId;
use util::nodemap::{DefIdSet, FnvHashMap};

use std::hash::Hash;
use syntax::ast::NodeId;

// Accessibility levels, sorted in ascending order
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessLevel {
    // Exported items + items participating in various kinds of public interfaces,
    // but not directly nameable. For example, if function `fn f() -> T {...}` is
    // public, then type `T` is exported. Its values can be obtained by other crates
    // even if the type itseld is not nameable.
    // FIXME: Mostly unimplemented. Only `type` aliases export items currently.
    Reachable,
    // Public items + items accessible to other crates with help of `pub use` reexports
    Exported,
    // Items accessible to other crates directly, without help of reexports
    Public,
}

// Accessibility levels for reachable HIR nodes
#[derive(Clone)]
pub struct AccessLevels<Id = NodeId> {
    pub map: FnvHashMap<Id, AccessLevel>
}

impl<Id: Hash + Eq> AccessLevels<Id> {
    pub fn is_reachable(&self, id: Id) -> bool {
        self.map.contains_key(&id)
    }
    pub fn is_exported(&self, id: Id) -> bool {
        self.map.get(&id) >= Some(&AccessLevel::Exported)
    }
    pub fn is_public(&self, id: Id) -> bool {
        self.map.get(&id) >= Some(&AccessLevel::Public)
    }
}

impl<Id: Hash + Eq> Default for AccessLevels<Id> {
    fn default() -> Self {
        AccessLevels { map: Default::default() }
    }
}

/// A set containing all exported definitions from external crates.
/// The set does not contain any entries from local crates.
pub type ExternalExports = DefIdSet;

#[derive(Copy, Clone, Debug)]
pub enum LastPrivate {
    LastMod(PrivateDep),
    // `use` directives (imports) can refer to two separate definitions in the
    // type and value namespaces. We record here the last private node for each
    // and whether the import is in fact used for each.
    // If the Option<PrivateDep> fields are None, it means there is no definition
    // in that namespace.
    LastImport{value_priv: Option<PrivateDep>,
               value_used: ImportUse,
               type_priv: Option<PrivateDep>,
               type_used: ImportUse},
}

#[derive(Copy, Clone, Debug)]
pub enum PrivateDep {
    AllPublic,
    DependsOn(DefId),
}

// How an import is used.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ImportUse {
    Unused,       // The import is not used.
    Used,         // The import is used.
}

impl LastPrivate {
    pub fn or(self, other: LastPrivate) -> LastPrivate {
        match (self, other) {
            (me, LastMod(AllPublic)) => me,
            (_, other) => other,
        }
    }
}
