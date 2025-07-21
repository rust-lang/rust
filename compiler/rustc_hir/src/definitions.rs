//! For each definition, we track the following data. A definition
//! here is defined somewhat circularly as "something with a `DefId`",
//! but it generally corresponds to things like structs, enums, etc.
//! There are also some rather random cases (like const initializer
//! expressions) that are mostly just leftovers.

use std::fmt::{self, Write};
use std::hash::Hash;

use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::Hash64;
use rustc_index::IndexVec;
use rustc_macros::{Decodable, Encodable};
use rustc_span::{Symbol, kw, sym};
use tracing::{debug, instrument};

pub use crate::def_id::DefPathHash;
use crate::def_id::{CRATE_DEF_INDEX, CrateNum, DefIndex, LOCAL_CRATE, LocalDefId, StableCrateId};
use crate::def_path_hash_map::DefPathHashMap;

/// The `DefPathTable` maps `DefIndex`es to `DefKey`s and vice versa.
/// Internally the `DefPathTable` holds a tree of `DefKey`s, where each `DefKey`
/// stores the `DefIndex` of its parent.
/// There is one `DefPathTable` for each crate.
#[derive(Debug)]
pub struct DefPathTable {
    stable_crate_id: StableCrateId,
    index_to_key: IndexVec<DefIndex, DefKey>,
    // We do only store the local hash, as all the definitions are from the current crate.
    def_path_hashes: IndexVec<DefIndex, Hash64>,
    def_path_hash_to_index: DefPathHashMap,
}

impl DefPathTable {
    fn new(stable_crate_id: StableCrateId) -> DefPathTable {
        DefPathTable {
            stable_crate_id,
            index_to_key: Default::default(),
            def_path_hashes: Default::default(),
            def_path_hash_to_index: Default::default(),
        }
    }

    fn allocate(&mut self, key: DefKey, def_path_hash: DefPathHash) -> DefIndex {
        // Assert that all DefPathHashes correctly contain the local crate's StableCrateId.
        debug_assert_eq!(self.stable_crate_id, def_path_hash.stable_crate_id());
        let local_hash = def_path_hash.local_hash();

        let index = self.index_to_key.push(key);
        debug!("DefPathTable::insert() - {key:?} <-> {index:?}");

        self.def_path_hashes.push(local_hash);
        debug_assert!(self.def_path_hashes.len() == self.index_to_key.len());

        // Check for hash collisions of DefPathHashes. These should be
        // exceedingly rare.
        if let Some(existing) = self.def_path_hash_to_index.insert(&local_hash, &index) {
            let def_path1 = DefPath::make(LOCAL_CRATE, existing, |idx| self.def_key(idx));
            let def_path2 = DefPath::make(LOCAL_CRATE, index, |idx| self.def_key(idx));

            // Continuing with colliding DefPathHashes can lead to correctness
            // issues. We must abort compilation.
            //
            // The likelihood of such a collision is very small, so actually
            // running into one could be indicative of a poor hash function
            // being used.
            //
            // See the documentation for DefPathHash for more information.
            panic!(
                "found DefPathHash collision between {def_path1:#?} and {def_path2:#?}. \
                    Compilation cannot continue."
            );
        }

        index
    }

    #[inline(always)]
    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.index_to_key[index]
    }

    #[instrument(level = "trace", skip(self), ret)]
    #[inline(always)]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        let hash = self.def_path_hashes[index];
        DefPathHash::new(self.stable_crate_id, hash)
    }

    pub fn enumerated_keys_and_path_hashes(
        &self,
    ) -> impl Iterator<Item = (DefIndex, &DefKey, DefPathHash)> + ExactSizeIterator {
        self.index_to_key
            .iter_enumerated()
            .map(move |(index, key)| (index, key, self.def_path_hash(index)))
    }
}

#[derive(Debug)]
pub struct DisambiguatorState {
    next: UnordMap<(LocalDefId, DefPathData), u32>,
}

impl DisambiguatorState {
    pub fn new() -> Self {
        Self { next: Default::default() }
    }

    /// Creates a `DisambiguatorState` where the next allocated `(LocalDefId, DefPathData)` pair
    /// will have `index` as the disambiguator.
    pub fn with(def_id: LocalDefId, data: DefPathData, index: u32) -> Self {
        let mut this = Self::new();
        this.next.insert((def_id, data), index);
        this
    }
}

/// The definition table containing node definitions.
/// It holds the `DefPathTable` for `LocalDefId`s/`DefPath`s.
/// It also stores mappings to convert `LocalDefId`s to/from `HirId`s.
#[derive(Debug)]
pub struct Definitions {
    table: DefPathTable,
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

        // The new path is in the same crate as `parent`, and will contain the stable_crate_id.
        // Therefore, we only need to include information of the parent's local hash.
        parent.local_hash().hash(&mut hasher);

        let DisambiguatedDefPathData { ref data, disambiguator } = self.disambiguated_data;

        std::mem::discriminant(data).hash(&mut hasher);
        if let Some(name) = data.hashed_symbol() {
            // Get a stable hash by considering the symbol chars rather than
            // the symbol index.
            name.as_str().hash(&mut hasher);
        }

        disambiguator.hash(&mut hasher);

        let local_hash = hasher.finish();

        // Construct the new DefPathHash, making sure that the `crate_id`
        // portion of the hash is properly copied from the parent. This way the
        // `crate_id` part will be recursively propagated from the root to all
        // DefPathHashes in this DefPathTable.
        DefPathHash::new(parent.stable_crate_id(), local_hash)
    }

    #[inline]
    pub fn get_opt_name(&self) -> Option<Symbol> {
        self.disambiguated_data.data.get_opt_name()
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
    pub fn as_sym(&self, verbose: bool) -> Symbol {
        match self.data.name() {
            DefPathDataName::Named(name) => {
                if verbose && self.disambiguator != 0 {
                    Symbol::intern(&format!("{}#{}", name, self.disambiguator))
                } else {
                    name
                }
            }
            DefPathDataName::Anon { namespace } => {
                if let DefPathData::AnonAssocTy(method) = self.data {
                    Symbol::intern(&format!("{}::{{{}#{}}}", method, namespace, self.disambiguator))
                } else {
                    Symbol::intern(&format!("{{{}#{}}}", namespace, self.disambiguator))
                }
            }
        }
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
            write!(s, "::{}", component.as_sym(true)).unwrap();
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
            write!(s, "{}", component.as_sym(true)).unwrap();
        }

        s
    }
}

/// New variants should only be added in synchronization with `enum DefKind`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    /// The crate root (marker).
    CrateRoot,

    // Different kinds of items and item-like things:
    /// An impl.
    Impl,
    /// An `extern` block.
    ForeignMod,
    /// A `use` item.
    Use,
    /// A global asm item.
    GlobalAsm,
    /// Something in the type namespace.
    TypeNs(Symbol),
    /// Something in the value namespace.
    ValueNs(Symbol),
    /// Something in the macro namespace.
    MacroNs(Symbol),
    /// Something in the lifetime namespace.
    LifetimeNs(Symbol),
    /// A closure expression.
    Closure,

    // Subportions of items:
    /// Implicit constructor for a unit or tuple-like struct or enum variant.
    Ctor,
    /// A constant expression (see `{ast,hir}::AnonConst`).
    AnonConst,
    /// An existential `impl Trait` type node.
    /// Argument position `impl Trait` have a `TypeNs` with their pretty-printed name.
    OpaqueTy,
    /// Used for remapped captured lifetimes in an existential `impl Trait` type node.
    OpaqueLifetime(Symbol),
    /// An anonymous associated type from an RPITIT. The symbol refers to the name of the method
    /// that defined the type.
    AnonAssocTy(Symbol),
    /// A synthetic body for a coroutine's by-move body.
    SyntheticCoroutineBody,
    /// Additional static data referred to by a static.
    NestedStatic,
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

    /// Adds a root definition (no parent) and a few other reserved definitions.
    pub fn new(stable_crate_id: StableCrateId) -> Definitions {
        let key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0,
            },
        };

        // We want *both* halves of a DefPathHash to depend on the crate-id of the defining crate.
        // The crate-id can be more easily changed than the DefPath of an item, so, in the case of
        // a crate-local DefPathHash collision, the user can simply "roll the dice again" for all
        // DefPathHashes in the crate by changing the crate disambiguator (e.g. via bumping the
        // crate's version number).
        //
        // Children paths will only hash the local portion, and still inherit the change to the
        // root hash.
        let def_path_hash =
            DefPathHash::new(stable_crate_id, Hash64::new(stable_crate_id.as_u64()));

        // Create the root definition.
        let mut table = DefPathTable::new(stable_crate_id);
        let root = LocalDefId { local_def_index: table.allocate(key, def_path_hash) };
        assert_eq!(root.local_def_index, CRATE_DEF_INDEX);

        Definitions { table }
    }

    /// Creates a definition with a parent definition.
    /// If there are multiple definitions with the same DefPathData and the same parent, use
    /// `disambiguator` to differentiate them. Distinct `DisambiguatorState` instances are not
    /// guaranteed to generate unique disambiguators and should instead ensure that the `parent`
    /// and `data` pair is distinct from other instances.
    pub fn create_def(
        &mut self,
        parent: LocalDefId,
        data: DefPathData,
        disambiguator: &mut DisambiguatorState,
    ) -> LocalDefId {
        // We can't use `Debug` implementation for `LocalDefId` here, since it tries to acquire a
        // reference to `Definitions` and we're already holding a mutable reference.
        debug!(
            "create_def(parent={}, data={data:?})",
            self.def_path(parent).to_string_no_crate_verbose(),
        );

        // The root node must be created in `new()`.
        assert!(data != DefPathData::CrateRoot);

        // Find the next free disambiguator for this key.
        let disambiguator = {
            let next_disamb = disambiguator.next.entry((parent, data)).or_insert(0);
            let disambiguator = *next_disamb;
            *next_disamb = next_disamb.checked_add(1).expect("disambiguator overflow");
            disambiguator
        };
        let key = DefKey {
            parent: Some(parent.local_def_index),
            disambiguated_data: DisambiguatedDefPathData { data, disambiguator },
        };

        let parent_hash = self.table.def_path_hash(parent.local_def_index);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        debug!("create_def: after disambiguation, key = {:?}", key);

        // Create the definition.
        LocalDefId { local_def_index: self.table.allocate(key, def_path_hash) }
    }

    #[inline(always)]
    /// Returns `None` if the `DefPathHash` does not correspond to a `LocalDefId`
    /// in the current compilation session. This can legitimately happen if the
    /// `DefPathHash` is from a `DefId` in an upstream crate or, during incr. comp.,
    /// if the `DefPathHash` is from a previous compilation session and
    /// the def-path does not exist anymore.
    pub fn local_def_path_hash_to_def_id(&self, hash: DefPathHash) -> Option<LocalDefId> {
        debug_assert!(hash.stable_crate_id() == self.table.stable_crate_id);
        self.table
            .def_path_hash_to_index
            .get(&hash.local_hash())
            .map(|local_def_index| LocalDefId { local_def_index })
    }

    pub fn def_path_hash_to_def_index_map(&self) -> &DefPathHashMap {
        &self.table.def_path_hash_to_index
    }

    pub fn num_definitions(&self) -> usize {
        self.table.def_path_hashes.len()
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
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name)
            | OpaqueLifetime(name) => Some(name),

            Impl
            | ForeignMod
            | CrateRoot
            | Use
            | GlobalAsm
            | Closure
            | Ctor
            | AnonConst
            | OpaqueTy
            | AnonAssocTy(..)
            | SyntheticCoroutineBody
            | NestedStatic => None,
        }
    }

    fn hashed_symbol(&self) -> Option<Symbol> {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name) | AnonAssocTy(name)
            | OpaqueLifetime(name) => Some(name),

            Impl
            | ForeignMod
            | CrateRoot
            | Use
            | GlobalAsm
            | Closure
            | Ctor
            | AnonConst
            | OpaqueTy
            | SyntheticCoroutineBody
            | NestedStatic => None,
        }
    }

    pub fn name(&self) -> DefPathDataName {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name)
            | OpaqueLifetime(name) => DefPathDataName::Named(name),
            // Note that this does not show up in user print-outs.
            CrateRoot => DefPathDataName::Anon { namespace: kw::Crate },
            Impl => DefPathDataName::Anon { namespace: kw::Impl },
            ForeignMod => DefPathDataName::Anon { namespace: kw::Extern },
            Use => DefPathDataName::Anon { namespace: kw::Use },
            GlobalAsm => DefPathDataName::Anon { namespace: sym::global_asm },
            Closure => DefPathDataName::Anon { namespace: sym::closure },
            Ctor => DefPathDataName::Anon { namespace: sym::constructor },
            AnonConst => DefPathDataName::Anon { namespace: sym::constant },
            OpaqueTy => DefPathDataName::Anon { namespace: sym::opaque },
            AnonAssocTy(..) => DefPathDataName::Anon { namespace: sym::anon_assoc },
            SyntheticCoroutineBody => DefPathDataName::Anon { namespace: sym::synthetic },
            NestedStatic => DefPathDataName::Anon { namespace: sym::nested },
        }
    }
}

impl fmt::Display for DefPathData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            DefPathDataName::Named(name) => f.write_str(name.as_str()),
            // FIXME(#70334): this will generate legacy {{closure}}, {{impl}}, etc
            DefPathDataName::Anon { namespace } => write!(f, "{{{{{namespace}}}}}"),
        }
    }
}
