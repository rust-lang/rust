// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use hir::map::def_collector::DefCollector;
use rustc_data_structures::fnv::FnvHashMap;
use std::fmt::Write;
use std::hash::{Hash, Hasher, SipHasher};
use syntax::{ast, visit};
use syntax::parse::token::{self, InternedString};
use ty::TyCtxt;
use util::nodemap::NodeMap;

/// The definition table containing node definitions
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct DefPath {
    /// the path leading from the crate root to the item
    pub data: Vec<DisambiguatedDefPathData>,

    /// what krate root is this path relative to?
    pub krate: CrateNum,
}

impl DefPath {
    pub fn is_local(&self) -> bool {
        self.krate == LOCAL_CRATE
    }

    pub fn make<FN>(start_krate: CrateNum,
                    start_index: DefIndex,
                    mut get_key: FN) -> DefPath
        where FN: FnMut(DefIndex) -> DefKey
    {
        let mut krate = start_krate;
        let mut data = vec![];
        let mut index = Some(start_index);
        loop {
            debug!("DefPath::make: krate={:?} index={:?}", krate, index);
            let p = index.unwrap();
            let key = get_key(p);
            debug!("DefPath::make: key={:?}", key);
            match key.disambiguated_data.data {
                DefPathData::CrateRoot => {
                    assert!(key.parent.is_none());
                    break;
                }
                DefPathData::InlinedRoot(ref p) => {
                    assert!(key.parent.is_none());
                    assert!(!p.def_id.is_local());
                    data.extend(p.data.iter().cloned().rev());
                    krate = p.def_id.krate;
                    break;
                }
                _ => {
                    data.push(key.disambiguated_data);
                    index = key.parent;
                }
            }
        }
        data.reverse();
        DefPath { data: data, krate: krate }
    }

    pub fn to_string(&self, tcx: TyCtxt) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        s.push_str(&tcx.original_crate_name(self.krate));
        s.push_str("/");
        s.push_str(&tcx.crate_disambiguator(self.krate));

        for component in &self.data {
            write!(s,
                   "::{}[{}]",
                   component.data.as_interned_str(),
                   component.disambiguator)
                .unwrap();
        }

        s
    }

    pub fn deterministic_hash(&self, tcx: TyCtxt) -> u64 {
        let mut state = SipHasher::new();
        self.deterministic_hash_to(tcx, &mut state);
        state.finish()
    }

    pub fn deterministic_hash_to<H: Hasher>(&self, tcx: TyCtxt, state: &mut H) {
        tcx.original_crate_name(self.krate).hash(state);
        tcx.crate_disambiguator(self.krate).hash(state);
        self.data.hash(state);
    }
}

/// Root of an inlined item. We track the `DefPath` of the item within
/// the original crate but also its def-id. This is kind of an
/// augmented version of a `DefPath` that includes a `DefId`. This is
/// all sort of ugly but the hope is that inlined items will be going
/// away soon anyway.
///
/// Some of the constraints that led to the current approach:
///
/// - I don't want to have a `DefId` in the main `DefPath` because
///   that gets serialized for incr. comp., and when reloaded the
///   `DefId` is no longer valid. I'd rather maintain the invariant
///   that every `DefId` is valid, and a potentially outdated `DefId` is
///   represented as a `DefPath`.
///   - (We don't serialize def-paths from inlined items, so it's ok to have one here.)
/// - We need to be able to extract the def-id from inline items to
///   make the symbol name. In theory we could retrace it from the
///   data, but the metadata doesn't have the required indices, and I
///   don't want to write the code to create one just for this.
/// - It may be that we don't actually need `data` at all. We'll have
///   to see about that.
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct InlinedRootPath {
    pub data: Vec<DisambiguatedDefPathData>,
    pub def_id: DefId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    /// The crate root (marker)
    CrateRoot,
    /// An inlined root
    InlinedRoot(Box<InlinedRootPath>),

    // Catch-all for random DefId things like DUMMY_NODE_ID
    Misc,

    // Different kinds of items and item-like things:
    /// An impl
    Impl,
    /// Something in the type NS
    TypeNs(InternedString),
    /// Something in the value NS
    ValueNs(InternedString),
    /// A module declaration
    Module(InternedString),
    /// A macro rule
    MacroDef(InternedString),
    /// A closure expression
    ClosureExpr,

    // Subportions of items
    /// A type parameter (generic parameter)
    TypeParam(InternedString),
    /// A lifetime definition
    LifetimeDef(InternedString),
    /// A variant of a enum
    EnumVariant(InternedString),
    /// A struct field
    Field(InternedString),
    /// Implicit ctor for a tuple-like struct
    StructCtor,
    /// Initializer for a const
    Initializer,
    /// Pattern binding
    Binding(InternedString),
    /// An `impl Trait` type node.
    ImplTrait
}

impl Definitions {
    /// Create new empty definition map.
    pub fn new() -> Definitions {
        Definitions {
            data: vec![],
            key_map: FnvHashMap(),
            node_map: NodeMap(),
        }
    }

    pub fn collect(&mut self, krate: &ast::Crate) {
        let mut def_collector = DefCollector::root(self);
        visit::walk_crate(&mut def_collector, krate);
    }

    /// Get the number of definitions.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.data[index.as_usize()].key.clone()
    }

    pub fn def_index_for_def_key(&self, key: DefKey) -> Option<DefIndex> {
        self.key_map.get(&key).cloned()
    }

    /// Returns the path from the crate root to `index`. The root
    /// nodes are not included in the path (i.e., this will be an
    /// empty vector for the crate root). For an inlined item, this
    /// will be the path of the item in the external crate (but the
    /// path will begin with the path to the external crate).
    pub fn def_path(&self, index: DefIndex) -> DefPath {
        DefPath::make(LOCAL_CRATE, index, |p| self.def_key(p))
    }

    pub fn opt_def_index(&self, node: ast::NodeId) -> Option<DefIndex> {
        self.node_map.get(&node).cloned()
    }

    pub fn opt_local_def_id(&self, node: ast::NodeId) -> Option<DefId> {
        self.opt_def_index(node).map(DefId::local)
    }

    pub fn local_def_id(&self, node: ast::NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap()
    }

    pub fn as_local_node_id(&self, def_id: DefId) -> Option<ast::NodeId> {
        if def_id.krate == LOCAL_CRATE {
            assert!(def_id.index.as_usize() < self.data.len());
            Some(self.data[def_id.index.as_usize()].node_id)
        } else {
            None
        }
    }

    /// Add a definition with a parent definition.
    pub fn create_def_with_parent(&mut self,
                                  parent: Option<DefIndex>,
                                  node_id: ast::NodeId,
                                  data: DefPathData)
                                  -> DefIndex {
        debug!("create_def_with_parent(parent={:?}, node_id={:?}, data={:?})",
               parent, node_id, data);

        assert!(!self.node_map.contains_key(&node_id),
                "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
                node_id,
                data,
                self.data[self.node_map[&node_id].as_usize()]);

        assert!(parent.is_some() ^ match data {
            DefPathData::CrateRoot | DefPathData::InlinedRoot(_) => true,
            _ => false,
        });

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

        debug!("create_def_with_parent: after disambiguation, key = {:?}", key);

        // Create the definition.
        let index = DefIndex::new(self.data.len());
        self.data.push(DefData { key: key.clone(), node_id: node_id });
        debug!("create_def_with_parent: node_map[{:?}] = {:?}", node_id, index);
        self.node_map.insert(node_id, index);
        debug!("create_def_with_parent: key_map[{:?}] = {:?}", key, index);
        self.key_map.insert(key, index);


        index
    }
}

impl DefPathData {
    pub fn get_opt_name(&self) -> Option<ast::Name> {
        use self::DefPathData::*;
        match *self {
            TypeNs(ref name) |
            ValueNs(ref name) |
            Module(ref name) |
            MacroDef(ref name) |
            TypeParam(ref name) |
            LifetimeDef(ref name) |
            EnumVariant(ref name) |
            Binding(ref name) |
            Field(ref name) => Some(token::intern(name)),

            Impl |
            CrateRoot |
            InlinedRoot(_) |
            Misc |
            ClosureExpr |
            StructCtor |
            Initializer |
            ImplTrait => None
        }
    }

    pub fn as_interned_str(&self) -> InternedString {
        use self::DefPathData::*;
        match *self {
            TypeNs(ref name) |
            ValueNs(ref name) |
            Module(ref name) |
            MacroDef(ref name) |
            TypeParam(ref name) |
            LifetimeDef(ref name) |
            EnumVariant(ref name) |
            Binding(ref name) |
            Field(ref name) => {
                name.clone()
            }

            Impl => {
                InternedString::new("{{impl}}")
            }

            // note that this does not show up in user printouts
            CrateRoot => {
                InternedString::new("{{root}}")
            }

            // note that this does not show up in user printouts
            InlinedRoot(_) => {
                InternedString::new("{{inlined-root}}")
            }

            Misc => {
                InternedString::new("{{?}}")
            }

            ClosureExpr => {
                InternedString::new("{{closure}}")
            }

            StructCtor => {
                InternedString::new("{{constructor}}")
            }

            Initializer => {
                InternedString::new("{{initializer}}")
            }

            ImplTrait => {
                InternedString::new("{{impl-Trait}}")
            }
        }
    }

    pub fn to_string(&self) -> String {
        self.as_interned_str().to_string()
    }
}

