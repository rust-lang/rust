//! For each definition, we track the following data. A definition
//! here is defined somewhat circularly as "something with a `DefId`",
//! but it generally corresponds to things like structs, enums, etc.
//! There are also some rather random cases (like const initializer
//! expressions) that are mostly just leftovers.

pub use crate::def_id::DefPathHash;
use crate::def_id::{CrateNum, DefIndex, LocalDefId, StableCrateId, CRATE_DEF_INDEX, LOCAL_CRATE};
use crate::hir;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::unhash::UnhashMap;
use rustc_index::vec::IndexVec;
use rustc_span::hygiene::ExpnId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use std::fmt::{self, Write};
use std::hash::Hash;
use tracing::debug;

/// The `DefPathTable` maps `DefIndex`es to `DefKey`s and vice versa.
/// Internally the `DefPathTable` holds a tree of `DefKey`s, where each `DefKey`
/// stores the `DefIndex` of its parent.
/// There is one `DefPathTable` for each crate.
#[derive(Clone, Default, Debug)]
pub struct DefPathTable {
    index_to_key: IndexVec<DefIndex, DefKey>,
    def_path_hashes: IndexVec<DefIndex, DefPathHash>,
    def_path_hash_to_index: UnhashMap<DefPathHash, DefIndex>,
}

impl DefPathTable {
    fn allocate(&mut self, key: DefKey, def_path_hash: DefPathHash) -> DefIndex {
        let index = {
            let index = DefIndex::from(self.index_to_key.len());
            debug!("DefPathTable::insert() - {:?} <-> {:?}", key, index);
            self.index_to_key.push(key);
            index
        };
        self.def_path_hashes.push(def_path_hash);
        debug_assert!(self.def_path_hashes.len() == self.index_to_key.len());

        // Check for hash collisions of DefPathHashes. These should be
        // exceedingly rare.
        if let Some(existing) = self.def_path_hash_to_index.insert(def_path_hash, index) {
            let def_path1 = DefPath::make(LOCAL_CRATE, existing, |idx| self.def_key(idx));
            let def_path2 = DefPath::make(LOCAL_CRATE, index, |idx| self.def_key(idx));

            // Continuing with colliding DefPathHashes can lead to correctness
            // issues. We must abort compilation.
            //
            // The likelyhood of such a collision is very small, so actually
            // running into one could be indicative of a poor hash function
            // being used.
            //
            // See the documentation for DefPathHash for more information.
            panic!(
                "found DefPathHash collsion between {:?} and {:?}. \
                    Compilation cannot continue.",
                def_path1, def_path2
            );
        }

        // Assert that all DefPathHashes correctly contain the local crate's
        // StableCrateId
        #[cfg(debug_assertions)]
        if let Some(root) = self.def_path_hashes.get(CRATE_DEF_INDEX) {
            assert!(def_path_hash.stable_crate_id() == root.stable_crate_id());
        }

        index
    }

    #[inline(always)]
    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.index_to_key[index]
    }

    #[inline(always)]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        let hash = self.def_path_hashes[index];
        debug!("def_path_hash({:?}) = {:?}", index, hash);
        hash
    }

    pub fn enumerated_keys_and_path_hashes(
        &self,
    ) -> impl Iterator<Item = (DefIndex, &DefKey, &DefPathHash)> + '_ {
        self.index_to_key
            .iter_enumerated()
            .map(move |(index, key)| (index, key, &self.def_path_hashes[index]))
    }
}

/// The definition table containing node definitions.
/// It holds the `DefPathTable` for `LocalDefId`s/`DefPath`s.
/// It also stores mappings to convert `LocalDefId`s to/from `HirId`s.
#[derive(Clone, Debug)]
pub struct Definitions {
    table: DefPathTable,

    // FIXME(eddyb) ideally all `LocalDefId`s would be HIR owners.
    pub(super) def_id_to_hir_id: IndexVec<LocalDefId, Option<hir::HirId>>,
    /// The reverse mapping of `def_id_to_hir_id`.
    pub(super) hir_id_to_def_id: FxHashMap<hir::HirId, LocalDefId>,

    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    expansions_that_defined: FxHashMap<LocalDefId, ExpnId>,

    def_id_to_span: IndexVec<LocalDefId, Span>,
}

/// A unique identifier that we can use to lookup a definition
/// precisely. It combines the index of the definition's parent (if
/// any) with a `DisambiguatedDefPathData`.
#[derive(Copy, Clone, PartialEq, Debug, Encodable, Decodable)]
pub struct DefKey {
    /// The parent path.
    pub parent: Option<DefIndex>,

    /// The identifier of this node.
    pub disambiguated_data: DisambiguatedDefPathData,
}

impl DefKey {
    pub(crate) fn compute_stable_hash(&self, parent: DefPathHash) -> DefPathHash {
        let mut hasher = StableHasher::new();

        parent.hash(&mut hasher);

        let DisambiguatedDefPathData { ref data, disambiguator } = self.disambiguated_data;

        std::mem::discriminant(data).hash(&mut hasher);
        if let Some(name) = data.get_opt_name() {
            // Get a stable hash by considering the symbol chars rather than
            // the symbol index.
            name.as_str().hash(&mut hasher);
        }

        disambiguator.hash(&mut hasher);

        let local_hash: u64 = hasher.finish();

        // Construct the new DefPathHash, making sure that the `crate_id`
        // portion of the hash is properly copied from the parent. This way the
        // `crate_id` part will be recursively propagated from the root to all
        // DefPathHashes in this DefPathTable.
        DefPathHash::new(parent.stable_crate_id(), local_hash)
    }
}

/// A pair of `DefPathData` and an integer disambiguator. The integer is
/// normally `0`, but in the event that there are multiple defs with the
/// same `parent` and `data`, we use this field to disambiguate
/// between them. This introduces some artificial ordering dependency
/// but means that if you have, e.g., two impls for the same type in
/// the same module, they do get distinct `DefId`s.
#[derive(Copy, Clone, PartialEq, Debug, Encodable, Decodable)]
pub struct DisambiguatedDefPathData {
    pub data: DefPathData,
    pub disambiguator: u32,
}

impl DisambiguatedDefPathData {
    pub fn fmt_maybe_verbose(&self, writer: &mut impl Write, verbose: bool) -> fmt::Result {
        match self.data.name() {
            DefPathDataName::Named(name) => {
                if verbose && self.disambiguator != 0 {
                    write!(writer, "{}#{}", name, self.disambiguator)
                } else {
                    writer.write_str(&name.as_str())
                }
            }
            DefPathDataName::Anon { namespace } => {
                write!(writer, "{{{}#{}}}", namespace, self.disambiguator)
            }
        }
    }
}

impl fmt::Display for DisambiguatedDefPathData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_maybe_verbose(f, true)
    }
}

#[derive(Clone, Debug, Encodable, Decodable)]
pub struct DefPath {
    /// The path leading from the crate root to the item.
    pub data: Vec<DisambiguatedDefPathData>,

    /// The crate root this path is relative to.
    pub krate: CrateNum,
}

impl DefPath {
    pub fn make<FN>(krate: CrateNum, start_index: DefIndex, mut get_key: FN) -> DefPath
    where
        FN: FnMut(DefIndex) -> DefKey,
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
        DefPath { data, krate }
    }

    /// Returns a string representation of the `DefPath` without
    /// the crate-prefix. This method is useful if you don't have
    /// a `TyCtxt` available.
    pub fn to_string_no_crate_verbose(&self) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        for component in &self.data {
            write!(s, "::{}", component).unwrap();
        }

        s
    }

    /// Returns a filename-friendly string of the `DefPath`, without
    /// the crate-prefix. This method is useful if you don't have
    /// a `TyCtxt` available.
    pub fn to_filename_friendly_no_crate(&self) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        let mut opt_delimiter = None;
        for component in &self.data {
            s.extend(opt_delimiter);
            opt_delimiter = Some('-');
            write!(s, "{}", component).unwrap();
        }

        s
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    /// The crate root (marker).
    CrateRoot,
    // Catch-all for random `DefId` things like `DUMMY_NODE_ID`.
    Misc,

    // Different kinds of items and item-like things:
    /// An impl.
    Impl,
    /// Something in the type namespace.
    TypeNs(Symbol),
    /// Something in the value namespace.
    ValueNs(Symbol),
    /// Something in the macro namespace.
    MacroNs(Symbol),
    /// Something in the lifetime namespace.
    LifetimeNs(Symbol),
    /// A closure expression.
    ClosureExpr,

    // Subportions of items:
    /// Implicit constructor for a unit or tuple-like struct or enum variant.
    Ctor,
    /// A constant expression (see `{ast,hir}::AnonConst`).
    AnonConst,
    /// An `impl Trait` type node.
    ImplTrait,
}

impl Definitions {
    pub fn def_path_table(&self) -> &DefPathTable {
        &self.table
    }

    /// Gets the number of definitions.
    pub fn def_index_count(&self) -> usize {
        self.table.index_to_key.len()
    }

    #[inline]
    pub fn def_key(&self, id: LocalDefId) -> DefKey {
        self.table.def_key(id.local_def_index)
    }

    #[inline(always)]
    pub fn def_path_hash(&self, id: LocalDefId) -> DefPathHash {
        self.table.def_path_hash(id.local_def_index)
    }

    /// Returns the path from the crate root to `index`. The root
    /// nodes are not included in the path (i.e., this will be an
    /// empty vector for the crate root). For an inlined item, this
    /// will be the path of the item in the external crate (but the
    /// path will begin with the path to the external crate).
    pub fn def_path(&self, id: LocalDefId) -> DefPath {
        DefPath::make(LOCAL_CRATE, id.local_def_index, |index| {
            self.def_key(LocalDefId { local_def_index: index })
        })
    }

    #[inline]
    #[track_caller]
    pub fn local_def_id_to_hir_id(&self, id: LocalDefId) -> hir::HirId {
        self.def_id_to_hir_id[id].unwrap()
    }

    #[inline]
    pub fn opt_hir_id_to_local_def_id(&self, hir_id: hir::HirId) -> Option<LocalDefId> {
        self.hir_id_to_def_id.get(&hir_id).copied()
    }

    /// Adds a root definition (no parent) and a few other reserved definitions.
    pub fn new(stable_crate_id: StableCrateId, crate_span: Span) -> Definitions {
        let key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0,
            },
        };

        let parent_hash = DefPathHash::new(stable_crate_id, 0);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        // Create the root definition.
        let mut table = DefPathTable::default();
        let root = LocalDefId { local_def_index: table.allocate(key, def_path_hash) };
        assert_eq!(root.local_def_index, CRATE_DEF_INDEX);

        let mut def_id_to_span = IndexVec::new();
        // A relative span's parent must be an absolute span.
        debug_assert_eq!(crate_span.data_untracked().parent, None);
        let _root = def_id_to_span.push(crate_span);
        debug_assert_eq!(_root, root);

        Definitions {
            table,
            def_id_to_hir_id: Default::default(),
            hir_id_to_def_id: Default::default(),
            expansions_that_defined: Default::default(),
            def_id_to_span,
        }
    }

    /// Retrieves the root definition.
    pub fn get_root_def(&self) -> LocalDefId {
        LocalDefId { local_def_index: CRATE_DEF_INDEX }
    }

    /// Adds a definition with a parent definition.
    pub fn create_def(
        &mut self,
        parent: LocalDefId,
        data: DefPathData,
        expn_id: ExpnId,
        mut next_disambiguator: impl FnMut(LocalDefId, DefPathData) -> u32,
        span: Span,
    ) -> LocalDefId {
        debug!("create_def(parent={:?}, data={:?}, expn_id={:?})", parent, data, expn_id);

        // The root node must be created with `create_root_def()`.
        assert!(data != DefPathData::CrateRoot);

        let disambiguator = next_disambiguator(parent, data);
        let key = DefKey {
            parent: Some(parent.local_def_index),
            disambiguated_data: DisambiguatedDefPathData { data, disambiguator },
        };

        let parent_hash = self.table.def_path_hash(parent.local_def_index);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        debug!("create_def: after disambiguation, key = {:?}", key);

        // Create the definition.
        let def_id = LocalDefId { local_def_index: self.table.allocate(key, def_path_hash) };

        if expn_id != ExpnId::root() {
            self.expansions_that_defined.insert(def_id, expn_id);
        }

        // A relative span's parent must be an absolute span.
        debug_assert_eq!(span.data_untracked().parent, None);
        let _id = self.def_id_to_span.push(span);
        debug_assert_eq!(_id, def_id);

        def_id
    }

    /// Initializes the `LocalDefId` to `HirId` mapping once it has been generated during
    /// AST to HIR lowering.
    pub fn init_def_id_to_hir_id_mapping(
        &mut self,
        mapping: IndexVec<LocalDefId, Option<hir::HirId>>,
    ) {
        assert!(
            self.def_id_to_hir_id.is_empty(),
            "trying to initialize `LocalDefId` <-> `HirId` mappings twice"
        );

        // Build the reverse mapping of `def_id_to_hir_id`.
        self.hir_id_to_def_id = mapping
            .iter_enumerated()
            .filter_map(|(def_id, hir_id)| hir_id.map(|hir_id| (hir_id, def_id)))
            .collect();

        self.def_id_to_hir_id = mapping;
    }

    pub fn expansion_that_defined(&self, id: LocalDefId) -> ExpnId {
        self.expansions_that_defined.get(&id).copied().unwrap_or_else(ExpnId::root)
    }

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate.
    #[inline]
    pub fn def_span(&self, def_id: LocalDefId) -> Span {
        self.def_id_to_span[def_id]
    }

    pub fn iter_local_def_id(&self) -> impl Iterator<Item = LocalDefId> + '_ {
        self.def_id_to_hir_id.iter_enumerated().map(|(k, _)| k)
    }

    #[inline(always)]
    pub fn local_def_path_hash_to_def_id(&self, hash: DefPathHash) -> Option<LocalDefId> {
        self.table
            .def_path_hash_to_index
            .get(&hash)
            .map(|&local_def_index| LocalDefId { local_def_index })
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DefPathDataName {
    Named(Symbol),
    Anon { namespace: Symbol },
}

impl DefPathData {
    pub fn get_opt_name(&self) -> Option<Symbol> {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name) => Some(name),

            Impl | CrateRoot | Misc | ClosureExpr | Ctor | AnonConst | ImplTrait => None,
        }
    }

    pub fn name(&self) -> DefPathDataName {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name) => {
                DefPathDataName::Named(name)
            }
            // Note that this does not show up in user print-outs.
            CrateRoot => DefPathDataName::Anon { namespace: kw::Crate },
            Impl => DefPathDataName::Anon { namespace: kw::Impl },
            Misc => DefPathDataName::Anon { namespace: sym::misc },
            ClosureExpr => DefPathDataName::Anon { namespace: sym::closure },
            Ctor => DefPathDataName::Anon { namespace: sym::constructor },
            AnonConst => DefPathDataName::Anon { namespace: sym::constant },
            ImplTrait => DefPathDataName::Anon { namespace: sym::opaque },
        }
    }
}

impl fmt::Display for DefPathData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            DefPathDataName::Named(name) => f.write_str(&name.as_str()),
            // FIXME(#70334): this will generate legacy {{closure}}, {{impl}}, etc
            DefPathDataName::Anon { namespace } => write!(f, "{{{{{}}}}}", namespace),
        }
    }
}
