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
use rustc_data_structures::indexed_vec::{IndexVec};
use rustc_data_structures::stable_hasher::StableHasher;
use serialize::{Encodable, Decodable, Encoder, Decoder};
use session::CrateDisambiguator;
use std::fmt::Write;
use std::hash::Hash;
use syntax::ast;
use syntax::ext::hygiene::Mark;
use syntax::symbol::{Symbol, InternedString};
use util::nodemap::NodeMap;

/// The DefPathTable maps DefIndexes to DefKeys and vice versa.
/// Internally the DefPathTable holds a tree of DefKeys, where each DefKey
/// stores the DefIndex of its parent.
/// There is one DefPathTable for each crate.
pub struct DefPathTable {
    index_to_key: [Vec<DefKey>; 2],
    def_path_hashes: [Vec<DefPathHash>; 2],
}

// Unfortunately we have to provide a manual impl of Clone because of the
// fixed-sized array field.
impl Clone for DefPathTable {
    fn clone(&self) -> Self {
        DefPathTable {
            index_to_key: [self.index_to_key[0].clone(),
                           self.index_to_key[1].clone()],
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
            let index = DefIndex::from_array_index(index_to_key.len(), address_space);
            debug!("DefPathTable::insert() - {:?} <-> {:?}", key, index);
            index_to_key.push(key);
            index
        };
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
        let ret = self.def_path_hashes[index.address_space().index()]
                                      [index.as_array_index()];
        debug!("def_path_hash({:?}) = {:?}", index, ret);
        return ret
    }

    pub fn add_def_path_hashes_to(&self,
                                  cnum: CrateNum,
                                  out: &mut FxHashMap<DefPathHash, DefId>) {
        for &address_space in &[DefIndexAddressSpace::Low, DefIndexAddressSpace::High] {
            out.extend(
                (&self.def_path_hashes[address_space.index()])
                    .iter()
                    .enumerate()
                    .map(|(index, &hash)| {
                        let def_id = DefId {
                            krate: cnum,
                            index: DefIndex::from_array_index(index, address_space),
                        };
                        (hash, def_id)
                    })
            );
        }
    }

    pub fn size(&self) -> usize {
        self.index_to_key.iter().map(|v| v.len()).sum()
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

        Ok(DefPathTable {
            index_to_key,
            def_path_hashes,
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
    next_disambiguator: FxHashMap<(DefIndex, DefPathData), u32>,
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
            next_disambiguator: self.next_disambiguator.clone(),
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

        let DisambiguatedDefPathData {
            ref data,
            disambiguator,
        } = self.disambiguated_data;

        ::std::mem::discriminant(data).hash(&mut hasher);
        match *data {
            DefPathData::TypeNs(name) |
            DefPathData::ValueNs(name) |
            DefPathData::Module(name) |
            DefPathData::MacroDef(name) |
            DefPathData::TypeParam(name) |
            DefPathData::LifetimeDef(name) |
            DefPathData::EnumVariant(name) |
            DefPathData::Field(name) |
            DefPathData::GlobalMetaData(name) => {
                name.hash(&mut hasher);
            }

            DefPathData::Impl |
            DefPathData::CrateRoot |
            DefPathData::Misc |
            DefPathData::ClosureExpr |
            DefPathData::StructCtor |
            DefPathData::Initializer |
            DefPathData::ImplTrait |
            DefPathData::Typeof => {}
        };

        disambiguator.hash(&mut hasher);

        DefPathHash(hasher.finish())
    }

    fn root_parent_stable_hash(crate_name: &str,
                               crate_disambiguator: CrateDisambiguator)
                               -> DefPathHash {
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

    /// Return filename friendly string of the DefPah without
    /// the crate-prefix. This method is useful if you don't have
    /// a TyCtxt available.
    pub fn to_filename_friendly_no_crate(&self) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        let mut opt_delimiter = None;
        for component in &self.data {
            opt_delimiter.map(|d| s.push(d));
            opt_delimiter = Some('-');
            if component.disambiguator == 0 {
                write!(s, "{}", component.data.as_interned_str()).unwrap();
            } else {
                write!(s,
                       "{}[{}]",
                       component.data.as_interned_str(),
                       component.disambiguator)
                    .unwrap();
            }
        }
        s
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, RustcEncodable, RustcDecodable)]
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
    /// An `impl Trait` type node.
    ImplTrait,
    /// A `typeof` type node.
    Typeof,

    /// GlobalMetaData identifies a piece of crate metadata that is global to
    /// a whole crate (as opposed to just one item). GlobalMetaData components
    /// are only supposed to show up right below the crate root.
    GlobalMetaData(InternedString)
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
                def_path_hashes: [vec![], vec![]],
            },
            node_to_def_index: NodeMap(),
            def_index_to_node: [vec![], vec![]],
            node_to_hir_id: IndexVec::new(),
            macro_def_scopes: FxHashMap(),
            expansions: FxHashMap(),
            next_disambiguator: FxHashMap(),
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

    /// Returns the path from the crate root to `index`. The root
    /// nodes are not included in the path (i.e., this will be an
    /// empty vector for the crate root). For an inlined item, this
    /// will be the path of the item in the external crate (but the
    /// path will begin with the path to the external crate).
    pub fn def_path(&self, index: DefIndex) -> DefPath {
        DefPath::make(LOCAL_CRATE, index, |p| self.def_key(p))
    }

    #[inline]
    pub fn opt_def_index(&self, node: ast::NodeId) -> Option<DefIndex> {
        self.node_to_def_index.get(&node).cloned()
    }

    #[inline]
    pub fn opt_local_def_id(&self, node: ast::NodeId) -> Option<DefId> {
        self.opt_def_index(node).map(DefId::local)
    }

    #[inline]
    pub fn local_def_id(&self, node: ast::NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap()
    }

    #[inline]
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

    #[inline]
    pub fn node_to_hir_id(&self, node_id: ast::NodeId) -> hir::HirId {
        self.node_to_hir_id[node_id]
    }

    pub fn find_node_for_hir_id(&self, hir_id: hir::HirId) -> ast::NodeId {
        self.node_to_hir_id
            .iter()
            .position(|x| *x == hir_id)
            .map(|idx| ast::NodeId::new(idx))
            .unwrap()
    }

    #[inline]
    pub fn def_index_to_hir_id(&self, def_index: DefIndex) -> hir::HirId {
        let space_index = def_index.address_space().index();
        let array_index = def_index.as_array_index();
        let node_id = self.def_index_to_node[space_index][array_index];
        self.node_to_hir_id[node_id]
    }

    /// Add a definition with a parent definition.
    pub fn create_root_def(&mut self,
                           crate_name: &str,
                           crate_disambiguator: CrateDisambiguator)
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

        // Find the next free disambiguator for this key.
        let disambiguator = {
            let next_disamb = self.next_disambiguator.entry((parent, data.clone())).or_insert(0);
            let disambiguator = *next_disamb;
            *next_disamb = next_disamb.checked_add(1).expect("disambiguator overflow");
            disambiguator
        };

        let key = DefKey {
            parent: Some(parent),
            disambiguated_data: DisambiguatedDefPathData {
                data, disambiguator
            }
        };

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

        let expansion = expansion.modern();
        if expansion != Mark::root() {
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
    pub fn get_opt_name(&self) -> Option<InternedString> {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) |
            ValueNs(name) |
            Module(name) |
            MacroDef(name) |
            TypeParam(name) |
            LifetimeDef(name) |
            EnumVariant(name) |
            Field(name) |
            GlobalMetaData(name) => Some(name),

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

    pub fn as_interned_str(&self) -> InternedString {
        use self::DefPathData::*;
        let s = match *self {
            TypeNs(name) |
            ValueNs(name) |
            Module(name) |
            MacroDef(name) |
            TypeParam(name) |
            LifetimeDef(name) |
            EnumVariant(name) |
            Field(name) |
            GlobalMetaData(name) => {
                return name
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

        const GLOBAL_MD_ADDRESS_SPACE: DefIndexAddressSpace = DefIndexAddressSpace::High;

        impl GlobalMetaDataKind {
            fn allocate_def_indices(definitions: &mut Definitions) {
                $({
                    let instance = GlobalMetaDataKind::$variant;
                    definitions.create_def_with_parent(
                        CRATE_DEF_INDEX,
                        ast::DUMMY_NODE_ID,
                        DefPathData::GlobalMetaData(instance.name().as_str()),
                        GLOBAL_MD_ADDRESS_SPACE,
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
                        data: DefPathData::GlobalMetaData(self.name().as_str()),
                        disambiguator: 0,
                    }
                };

                // These DefKeys are all right after the root,
                // so a linear search is fine.
                let index = def_path_table.index_to_key[GLOBAL_MD_ADDRESS_SPACE.index()]
                                          .iter()
                                          .position(|k| *k == def_key)
                                          .unwrap();

                DefIndex::from_array_index(index, GLOBAL_MD_ADDRESS_SPACE)
            }

            fn name(&self) -> Symbol {

                let string = match *self {
                    $(
                        GlobalMetaDataKind::$variant => {
                            concat!("{{GlobalMetaData::", stringify!($variant), "}}")
                        }
                    )*
                };

                Symbol::intern(string)
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
