// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For each definition, we track the following data.  A definition
//! here is defined somewhat circularly as "something with a def-id",
//! but it generally corresponds to things like structs, enums, etc.
//! There are also some rather random cases (like const initializer
//! expressions) that are mostly just leftovers.

use hir;
use hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE, DefIndexAddressSpace,
                  CRATE_DEF_INDEX};
use ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::stable_hasher::StableHasher;
use serialize::{Encodable, Decodable, Encoder, Decoder};
use std::fmt::Write;
use std::hash::Hash;
use syntax::ast::{self, Ident};
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::symbol::{Symbol, InternedString};
use ty::TyCtxt;
use util::nodemap::NodeMap;

/// The DefPathTable maps DefIndexes to DefKeys and vice versa.
/// Internally the DefPathTable holds a tree of DefKeys, where each DefKey
/// stores the DefIndex of its parent.
/// There is one DefPathTable for each crate.
pub struct DefPathTable {
    index_to_key: [Vec<DefKey>; 2],
    key_to_index: FxHashMap<DefKey, DefIndex>,
    def_path_hashes: [Vec<DefPathHash>; 2],
}

// Unfortunately we have to provide a manual impl of Clone because of the
// fixed-sized array field.
impl Clone for DefPathTable {
    fn clone(&self) -> Self {
        DefPathTable {
            index_to_key: [self.index_to_key[0].clone(),
                           self.index_to_key[1].clone()],
            key_to_index: self.key_to_index.clone(),
            def_path_hashes: [self.def_path_hashes[0].clone(),
                              self.def_path_hashes[1].clone()],
        }
    }
}

impl DefPathTable {

    fn allocate(&mut self,
                key: DefKey,
                def_path_hash: DefPathHash,
                address_space: DefIndexAddressSpace)
                -> DefIndex {
        let index = {
            let index_to_key = &mut self.index_to_key[address_space.index()];
            let index = DefIndex::new(index_to_key.len() + address_space.start());
            debug!("DefPathTable::insert() - {:?} <-> {:?}", key, index);
            index_to_key.push(key.clone());
            index
        };
        self.key_to_index.insert(key, index);
        self.def_path_hashes[address_space.index()].push(def_path_hash);
        debug_assert!(self.def_path_hashes[address_space.index()].len() ==
                      self.index_to_key[address_space.index()].len());
        index
    }

    #[inline(always)]
    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.index_to_key[index.address_space().index()]
                         [index.as_array_index()].clone()
    }

    #[inline(always)]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        self.def_path_hashes[index.address_space().index()]
                            [index.as_array_index()]
    }

    #[inline(always)]
    pub fn def_index_for_def_key(&self, key: &DefKey) -> Option<DefIndex> {
        self.key_to_index.get(key).cloned()
    }

    #[inline(always)]
    pub fn contains_key(&self, key: &DefKey) -> bool {
        self.key_to_index.contains_key(key)
    }

    pub fn retrace_path(&self,
                        path_data: &[DisambiguatedDefPathData])
                        -> Option<DefIndex> {
        let root_key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0,
            },
        };

        let root_index = self.key_to_index
                             .get(&root_key)
                             .expect("no root key?")
                             .clone();

        debug!("retrace_path: root_index={:?}", root_index);

        let mut index = root_index;
        for data in path_data {
            let key = DefKey { parent: Some(index), disambiguated_data: data.clone() };
            debug!("retrace_path: key={:?}", key);
            match self.key_to_index.get(&key) {
                Some(&i) => index = i,
                None => return None,
            }
        }

        Some(index)
    }

    pub fn add_def_path_hashes_to(&self,
                                  cnum: CrateNum,
                                  out: &mut FxHashMap<DefPathHash, DefId>) {
        for address_space in &[DefIndexAddressSpace::Low, DefIndexAddressSpace::High] {
            let start_index = address_space.start();
            out.extend(
                (&self.def_path_hashes[address_space.index()])
                    .iter()
                    .enumerate()
                    .map(|(index, &hash)| {
                        let def_id = DefId {
                            krate: cnum,
                            index: DefIndex::new(index + start_index),
                        };
                        (hash, def_id)
                    })
            );
        }
    }

    pub fn size(&self) -> usize {
        self.key_to_index.len()
    }
}


impl Encodable for DefPathTable {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        // Index to key
        self.index_to_key[DefIndexAddressSpace::Low.index()].encode(s)?;
        self.index_to_key[DefIndexAddressSpace::High.index()].encode(s)?;

        // DefPath hashes
        self.def_path_hashes[DefIndexAddressSpace::Low.index()].encode(s)?;
        self.def_path_hashes[DefIndexAddressSpace::High.index()].encode(s)?;

        Ok(())
    }
}

impl Decodable for DefPathTable {
    fn decode<D: Decoder>(d: &mut D) -> Result<DefPathTable, D::Error> {
        let index_to_key_lo: Vec<DefKey> = Decodable::decode(d)?;
        let index_to_key_hi: Vec<DefKey> = Decodable::decode(d)?;

        let def_path_hashes_lo: Vec<DefPathHash> = Decodable::decode(d)?;
        let def_path_hashes_hi: Vec<DefPathHash> = Decodable::decode(d)?;

        let index_to_key = [index_to_key_lo, index_to_key_hi];
        let def_path_hashes = [def_path_hashes_lo, def_path_hashes_hi];

        let mut key_to_index = FxHashMap();

        for space in &[DefIndexAddressSpace::Low, DefIndexAddressSpace::High] {
            key_to_index.extend(index_to_key[space.index()]
                .iter()
                .enumerate()
                .map(|(index, key)| (key.clone(),
                                     DefIndex::new(index + space.start()))))
        }

        Ok(DefPathTable {
            index_to_key: index_to_key,
            key_to_index: key_to_index,
            def_path_hashes: def_path_hashes,
        })
    }
}


/// The definition table containing node definitions.
/// It holds the DefPathTable for local DefIds/DefPaths and it also stores a
/// mapping from NodeIds to local DefIds.
pub struct Definitions {
    table: DefPathTable,
    node_to_def_index: NodeMap<DefIndex>,
    def_index_to_node: [Vec<ast::NodeId>; 2],
    pub(super) node_to_hir_id: IndexVec<ast::NodeId, hir::HirId>,
    macro_def_scopes: FxHashMap<Mark, DefId>,
    expansions: FxHashMap<DefIndex, Mark>,
}

// Unfortunately we have to provide a manual impl of Clone because of the
// fixed-sized array field.
impl Clone for Definitions {
    fn clone(&self) -> Self {
        Definitions {
            table: self.table.clone(),
            node_to_def_index: self.node_to_def_index.clone(),
            def_index_to_node: [
                self.def_index_to_node[0].clone(),
                self.def_index_to_node[1].clone(),
            ],
            node_to_hir_id: self.node_to_hir_id.clone(),
            macro_def_scopes: self.macro_def_scopes.clone(),
            expansions: self.expansions.clone(),
        }
    }
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

impl DefKey {
    fn compute_stable_hash(&self, parent_hash: DefPathHash) -> DefPathHash {
        let mut hasher = StableHasher::new();

        // We hash a 0u8 here to disambiguate between regular DefPath hashes,
        // and the special "root_parent" below.
        0u8.hash(&mut hasher);
        parent_hash.hash(&mut hasher);
        self.disambiguated_data.hash(&mut hasher);
        DefPathHash(hasher.finish())
    }

    fn root_parent_stable_hash(crate_name: &str, crate_disambiguator: &str) -> DefPathHash {
        let mut hasher = StableHasher::new();
        // Disambiguate this from a regular DefPath hash,
        // see compute_stable_hash() above.
        1u8.hash(&mut hasher);
        crate_name.hash(&mut hasher);
        crate_disambiguator.hash(&mut hasher);
        DefPathHash(hasher.finish())
    }
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

    pub fn make<FN>(krate: CrateNum,
                    start_index: DefIndex,
                    mut get_key: FN) -> DefPath
        where FN: FnMut(DefIndex) -> DefKey
    {
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

        s.push_str(&tcx.original_crate_name(self.krate).as_str());
        s.push_str("/");
        // Don't print the whole crate disambiguator. That's just annoying in
        // debug output.
        s.push_str(&tcx.crate_disambiguator(self.krate).as_str()[..7]);

        for component in &self.data {
            write!(s,
                   "::{}[{}]",
                   component.data.as_interned_str(),
                   component.disambiguator)
                .unwrap();
        }

        s
    }

    /// Returns a string representation of the DefPath without
    /// the crate-prefix. This method is useful if you don't have
    /// a TyCtxt available.
    pub fn to_string_no_crate(&self) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        for component in &self.data {
            write!(s,
                   "::{}[{}]",
                   component.data.as_interned_str(),
                   component.disambiguator)
                .unwrap();
        }

        s
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    /// The crate root (marker)
    CrateRoot,

    // Catch-all for random DefId things like DUMMY_NODE_ID
    Misc,

    // Different kinds of items and item-like things:
    /// An impl
    Impl,
    /// Something in the type NS
    TypeNs(Ident),
    /// Something in the value NS
    ValueNs(Ident),
    /// A module declaration
    Module(Ident),
    /// A macro rule
    MacroDef(Ident),
    /// A closure expression
    ClosureExpr,

    // Subportions of items
    /// A type parameter (generic parameter)
    TypeParam(Ident),
    /// A lifetime definition
    LifetimeDef(Ident),
    /// A variant of a enum
    EnumVariant(Ident),
    /// A struct field
    Field(Ident),
    /// Implicit ctor for a tuple-like struct
    StructCtor,
    /// Initializer for a const
    Initializer,
    /// Pattern binding
    Binding(Ident),
    /// An `impl Trait` type node.
    ImplTrait,
    /// A `typeof` type node.
    Typeof,

    /// GlobalMetaData identifies a piece of crate metadata that is global to
    /// a whole crate (as opposed to just one item). GlobalMetaData components
    /// are only supposed to show up right below the crate root.
    GlobalMetaData(Ident)
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug,
         RustcEncodable, RustcDecodable)]
pub struct DefPathHash(pub Fingerprint);

impl_stable_hash_for!(tuple_struct DefPathHash { fingerprint });

impl Definitions {
    /// Create new empty definition map.
    pub fn new() -> Definitions {
        Definitions {
            table: DefPathTable {
                index_to_key: [vec![], vec![]],
                key_to_index: FxHashMap(),
                def_path_hashes: [vec![], vec![]],
            },
            node_to_def_index: NodeMap(),
            def_index_to_node: [vec![], vec![]],
            node_to_hir_id: IndexVec::new(),
            macro_def_scopes: FxHashMap(),
            expansions: FxHashMap(),
        }
    }

    pub fn def_path_table(&self) -> &DefPathTable {
        &self.table
    }

    /// Get the number of definitions.
    pub fn def_index_counts_lo_hi(&self) -> (usize, usize) {
        (self.table.index_to_key[DefIndexAddressSpace::Low.index()].len(),
         self.table.index_to_key[DefIndexAddressSpace::High.index()].len())
    }

    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.table.def_key(index)
    }

    #[inline(always)]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        self.table.def_path_hash(index)
    }

    pub fn def_index_for_def_key(&self, key: DefKey) -> Option<DefIndex> {
        self.table.def_index_for_def_key(&key)
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
        self.node_to_def_index.get(&node).cloned()
    }

    pub fn opt_local_def_id(&self, node: ast::NodeId) -> Option<DefId> {
        self.opt_def_index(node).map(DefId::local)
    }

    pub fn local_def_id(&self, node: ast::NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap()
    }

    pub fn as_local_node_id(&self, def_id: DefId) -> Option<ast::NodeId> {
        if def_id.krate == LOCAL_CRATE {
            let space_index = def_id.index.address_space().index();
            let array_index = def_id.index.as_array_index();
            let node_id = self.def_index_to_node[space_index][array_index];
            if node_id != ast::DUMMY_NODE_ID {
                Some(node_id)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn node_to_hir_id(&self, node_id: ast::NodeId) -> hir::HirId {
        self.node_to_hir_id[node_id]
    }

    /// Add a definition with a parent definition.
    pub fn create_root_def(&mut self,
                           crate_name: &str,
                           crate_disambiguator: &str)
                           -> DefIndex {
        let key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0
            }
        };

        let parent_hash = DefKey::root_parent_stable_hash(crate_name,
                                                          crate_disambiguator);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        // Create the definition.
        let address_space = super::ITEM_LIKE_SPACE;
        let root_index = self.table.allocate(key, def_path_hash, address_space);
        assert_eq!(root_index, CRATE_DEF_INDEX);
        assert!(self.def_index_to_node[address_space.index()].is_empty());
        self.def_index_to_node[address_space.index()].push(ast::CRATE_NODE_ID);
        self.node_to_def_index.insert(ast::CRATE_NODE_ID, root_index);

        // Allocate some other DefIndices that always must exist.
        GlobalMetaDataKind::allocate_def_indices(self);

        root_index
    }

    /// Add a definition with a parent definition.
    pub fn create_def_with_parent(&mut self,
                                  parent: DefIndex,
                                  node_id: ast::NodeId,
                                  data: DefPathData,
                                  address_space: DefIndexAddressSpace,
                                  expansion: Mark)
                                  -> DefIndex {
        debug!("create_def_with_parent(parent={:?}, node_id={:?}, data={:?})",
               parent, node_id, data);

        assert!(!self.node_to_def_index.contains_key(&node_id),
                "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
                node_id,
                data,
                self.table.def_key(self.node_to_def_index[&node_id]));

        // The root node must be created with create_root_def()
        assert!(data != DefPathData::CrateRoot);

        // Find a unique DefKey. This basically means incrementing the disambiguator
        // until we get no match.
        let mut key = DefKey {
            parent: Some(parent),
            disambiguated_data: DisambiguatedDefPathData {
                data: data,
                disambiguator: 0
            }
        };

        while self.table.contains_key(&key) {
            key.disambiguated_data.disambiguator += 1;
        }

        let parent_hash = self.table.def_path_hash(parent);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        debug!("create_def_with_parent: after disambiguation, key = {:?}", key);

        // Create the definition.
        let index = self.table.allocate(key, def_path_hash, address_space);
        assert_eq!(index.as_array_index(),
                   self.def_index_to_node[address_space.index()].len());
        self.def_index_to_node[address_space.index()].push(node_id);

        // Some things for which we allocate DefIndices don't correspond to
        // anything in the AST, so they don't have a NodeId. For these cases
        // we don't need a mapping from NodeId to DefIndex.
        if node_id != ast::DUMMY_NODE_ID {
            debug!("create_def_with_parent: def_index_to_node[{:?} <-> {:?}", index, node_id);
            self.node_to_def_index.insert(node_id, index);
        }

        if expansion.is_modern() {
            self.expansions.insert(index, expansion);
        }

        index
    }

    /// Initialize the ast::NodeId to HirId mapping once it has been generated during
    /// AST to HIR lowering.
    pub fn init_node_id_to_hir_id_mapping(&mut self,
                                          mapping: IndexVec<ast::NodeId, hir::HirId>) {
        assert!(self.node_to_hir_id.is_empty(),
                "Trying initialize NodeId -> HirId mapping twice");
        self.node_to_hir_id = mapping;
    }

    pub fn expansion(&self, index: DefIndex) -> Mark {
        self.expansions.get(&index).cloned().unwrap_or(Mark::root())
    }

    pub fn macro_def_scope(&self, mark: Mark) -> DefId {
        self.macro_def_scopes[&mark]
    }

    pub fn add_macro_def_scope(&mut self, mark: Mark, scope: DefId) {
        self.macro_def_scopes.insert(mark, scope);
    }
}

impl DefPathData {
    pub fn get_opt_ident(&self) -> Option<Ident> {
        use self::DefPathData::*;
        match *self {
            TypeNs(ident) |
            ValueNs(ident) |
            Module(ident) |
            MacroDef(ident) |
            TypeParam(ident) |
            LifetimeDef(ident) |
            EnumVariant(ident) |
            Binding(ident) |
            Field(ident) |
            GlobalMetaData(ident) => Some(ident),

            Impl |
            CrateRoot |
            Misc |
            ClosureExpr |
            StructCtor |
            Initializer |
            ImplTrait |
            Typeof => None
        }
    }

    pub fn get_opt_name(&self) -> Option<ast::Name> {
        self.get_opt_ident().map(|ident| ident.name)
    }

    pub fn as_interned_str(&self) -> InternedString {
        use self::DefPathData::*;
        let s = match *self {
            TypeNs(ident) |
            ValueNs(ident) |
            Module(ident) |
            MacroDef(ident) |
            TypeParam(ident) |
            LifetimeDef(ident) |
            EnumVariant(ident) |
            Binding(ident) |
            Field(ident) |
            GlobalMetaData(ident) => {
                return ident.name.as_str();
            }

            // note that this does not show up in user printouts
            CrateRoot => "{{root}}",

            Impl => "{{impl}}",
            Misc => "{{?}}",
            ClosureExpr => "{{closure}}",
            StructCtor => "{{constructor}}",
            Initializer => "{{initializer}}",
            ImplTrait => "{{impl-Trait}}",
            Typeof => "{{typeof}}",
        };

        Symbol::intern(s).as_str()
    }

    pub fn to_string(&self) -> String {
        self.as_interned_str().to_string()
    }
}

impl Eq for DefPathData {}
impl PartialEq for DefPathData {
    fn eq(&self, other: &DefPathData) -> bool {
        ::std::mem::discriminant(self) == ::std::mem::discriminant(other) &&
        self.get_opt_ident() == other.get_opt_ident()
    }
}

impl ::std::hash::Hash for DefPathData {
    fn hash<H: ::std::hash::Hasher>(&self, hasher: &mut H) {
        ::std::mem::discriminant(self).hash(hasher);
        if let Some(ident) = self.get_opt_ident() {
            if ident.ctxt == SyntaxContext::empty() && ident.name == ident.name.interned() {
                ident.name.as_str().hash(hasher)
            } else {
                // FIXME(jseyfried) implement stable hashing for idents with macros 2.0 hygiene info
                ident.hash(hasher)
            }
        }
    }
}


// We define the GlobalMetaDataKind enum with this macro because we want to
// make sure that we exhaustively iterate over all variants when registering
// the corresponding DefIndices in the DefTable.
macro_rules! define_global_metadata_kind {
    (pub enum GlobalMetaDataKind {
        $($variant:ident),*
    }) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub enum GlobalMetaDataKind {
            $($variant),*
        }

        impl GlobalMetaDataKind {
            fn allocate_def_indices(definitions: &mut Definitions) {
                $({
                    let instance = GlobalMetaDataKind::$variant;
                    definitions.create_def_with_parent(
                        CRATE_DEF_INDEX,
                        ast::DUMMY_NODE_ID,
                        DefPathData::GlobalMetaData(instance.ident()),
                        DefIndexAddressSpace::High,
                        Mark::root()
                    );

                    // Make sure calling def_index does not crash.
                    instance.def_index(&definitions.table);
                })*
            }

            pub fn def_index(&self, def_path_table: &DefPathTable) -> DefIndex {
                let def_key = DefKey {
                    parent: Some(CRATE_DEF_INDEX),
                    disambiguated_data: DisambiguatedDefPathData {
                        data: DefPathData::GlobalMetaData(self.ident()),
                        disambiguator: 0,
                    }
                };

                def_path_table.key_to_index[&def_key]
            }

            fn ident(&self) -> Ident {

                let string = match *self {
                    $(
                        GlobalMetaDataKind::$variant => {
                            concat!("{{GlobalMetaData::", stringify!($variant), "}}")
                        }
                    )*
                };

                Ident::from_str(string)
            }
        }
    )
}

define_global_metadata_kind!(pub enum GlobalMetaDataKind {
    Krate,
    CrateDeps,
    DylibDependencyFormats,
    LangItems,
    LangItemsMissing,
    NativeLibraries,
    CodeMap,
    Impls,
    ExportedSymbols
});
