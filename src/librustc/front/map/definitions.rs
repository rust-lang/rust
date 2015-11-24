// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::cstore::LOCAL_CRATE;
use middle::def_id::{DefId, DefIndex};
use rustc_data_structures::fnv::FnvHashMap;
use rustc_front::hir;
use syntax::ast;
use syntax::parse::token::InternedString;
use util::nodemap::NodeMap;

#[derive(Clone)]
pub struct Definitions {
    data: Vec<DefData>,
    key_map: FnvHashMap<DefKey, DefIndex>,
    node_map: NodeMap<DefIndex>,
}

/// A unique identifier that we can use to lookup a definition
/// precisely. It combines the index of the definition's parent (if
/// any) with a `DisambiguatedDefPathData`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct DefKey {
    /// Parent path.
    pub parent: Option<DefIndex>,

    /// Identifier of this node.
    pub disambiguated_data: DisambiguatedDefPathData,
}

/// Pair of `DefPathData` and an integer disambiguator. The integer is
/// normally 0, but in the event that there are multiple defs with the
/// same `parent` and `data`, we use this field to disambiguate
/// between them. This introduces some artificial ordering dependency
/// but means that if you have (e.g.) two impls for the same type in
/// the same module, they do get distinct def-ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct DisambiguatedDefPathData {
    pub data: DefPathData,
    pub disambiguator: u32
}

/// For each definition, we track the following data.  A definition
/// here is defined somewhat circularly as "something with a def-id",
/// but it generally corresponds to things like structs, enums, etc.
/// There are also some rather random cases (like const initializer
/// expressions) that are mostly just leftovers.
#[derive(Clone, Debug)]
pub struct DefData {
    pub key: DefKey,

    /// Local ID within the HIR.
    pub node_id: ast::NodeId,
}

pub type DefPath = Vec<DisambiguatedDefPathData>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    CrateRoot,
    InlinedRoot(DefPath),

    // Catch-all for random DefId things like DUMMY_NODE_ID
    Misc,

    // Different kinds of items and item-like things:
    Impl,
    Type(ast::Name),
    Mod(ast::Name),
    Value(ast::Name),
    MacroDef(ast::Name),
    ClosureExpr,

    // Subportions of items
    TypeParam(ast::Name),
    LifetimeDef(ast::Name),
    EnumVariant(ast::Name),
    PositionalField,
    Field(hir::StructFieldKind),
    StructCtor, // implicit ctor for a tuple-like struct
    Initializer, // initializer for a const
    Binding(ast::Name), // pattern binding

    // An external crate that does not have an `extern crate` in this
    // crate.
    DetachedCrate(ast::Name),
}

impl Definitions {
    pub fn new() -> Definitions {
        Definitions {
            data: vec![],
            key_map: FnvHashMap(),
            node_map: NodeMap(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.data[index.as_usize()].key.clone()
    }

    /// Returns the path from the crate root to `index`. The root
    /// nodes are not included in the path (i.e., this will be an
    /// empty vector for the crate root). For an inlined item, this
    /// will be the path of the item in the external crate (but the
    /// path will begin with the path to the external crate).
    pub fn def_path(&self, index: DefIndex) -> DefPath {
        make_def_path(index, |p| self.def_key(p))
    }

    pub fn opt_def_index(&self, node: ast::NodeId) -> Option<DefIndex> {
        self.node_map.get(&node).cloned()
    }

    pub fn opt_local_def_id(&self, node: ast::NodeId) -> Option<DefId> {
        self.opt_def_index(node).map(DefId::local)
    }

    pub fn as_local_node_id(&self, def_id: DefId) -> Option<ast::NodeId> {
        if def_id.krate == LOCAL_CRATE {
            assert!(def_id.index.as_usize() < self.data.len());
            Some(self.data[def_id.index.as_usize()].node_id)
        } else {
            None
        }
    }

    pub fn create_def_with_parent(&mut self,
                                  parent: Option<DefIndex>,
                                  node_id: ast::NodeId,
                                  data: DefPathData)
                                  -> DefIndex {
        assert!(!self.node_map.contains_key(&node_id),
                "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
                node_id,
                data,
                self.data[self.node_map[&node_id].as_usize()]);

        // Find a unique DefKey. This basically means incrementing the disambiguator
        // until we get no match.
        let mut key = DefKey {
            parent: parent,
            disambiguated_data: DisambiguatedDefPathData {
                data: data,
                disambiguator: 0
            }
        };

        while self.key_map.contains_key(&key) {
            key.disambiguated_data.disambiguator += 1;
        }

        // Create the definition.
        let index = DefIndex::new(self.data.len());
        self.data.push(DefData { key: key.clone(), node_id: node_id });
        self.node_map.insert(node_id, index);
        self.key_map.insert(key, index);

        index
    }
}

impl DefPathData {
    pub fn as_interned_str(&self) -> InternedString {
        use self::DefPathData::*;
        match *self {
            Type(name) |
            Mod(name) |
            Value(name) |
            MacroDef(name) |
            TypeParam(name) |
            LifetimeDef(name) |
            EnumVariant(name) |
            DetachedCrate(name) |
            Binding(name) => {
                name.as_str()
            }

            Field(hir::StructFieldKind::NamedField(name, _)) => {
                name.as_str()
            }

            PositionalField |
            Field(hir::StructFieldKind::UnnamedField(_)) => {
                InternedString::new("<field>")
            }

            // note that this does not show up in user printouts
            CrateRoot => {
                InternedString::new("<root>")
            }

            // note that this does not show up in user printouts
            InlinedRoot(_) => {
                InternedString::new("<inlined-root>")
            }

            Misc => {
                InternedString::new("?")
            }

            Impl => {
                InternedString::new("<impl>")
            }

            ClosureExpr => {
                InternedString::new("<closure>")
            }

            StructCtor => {
                InternedString::new("<constructor>")
            }

            Initializer => {
                InternedString::new("<initializer>")
            }
        }
    }

    pub fn to_string(&self) -> String {
        self.as_interned_str().to_string()
    }
}

pub fn make_def_path<FN>(start_index: DefIndex, mut get_key: FN) -> DefPath
    where FN: FnMut(DefIndex) -> DefKey
{
    let mut result = vec![];
    let mut index = Some(start_index);
    while let Some(p) = index {
        let key = get_key(p);
        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                break;
            }
            DefPathData::InlinedRoot(ref p) => {
                assert!(key.parent.is_none());
                result.extend(p.iter().cloned().rev());
                break;
            }
            _ => {
                result.push(key.disambiguated_data);
                index = key.parent;
            }
        }
    }
    result.reverse();
    result
}
