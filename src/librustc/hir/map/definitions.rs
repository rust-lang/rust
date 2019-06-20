//! For each definition, we track the following data. A definition
//! here is defined somewhat circularly as "something with a `DefId`",
//! but it generally corresponds to things like structs, enums, etc.
//! There are also some rather random cases (like const initializer
//! expressions) that are mostly just leftovers.

use crate::hir;
use crate::hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE, CRATE_DEF_INDEX};
use crate::ich::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{IndexVec};
use rustc_data_structures::stable_hasher::StableHasher;
use crate::session::CrateDisambiguator;
use std::borrow::Borrow;
use std::fmt::Write;
use std::hash::Hash;
use syntax::ast;
use syntax::ext::hygiene::Mark;
use syntax::symbol::{Symbol, sym, InternedString};
use syntax_pos::{Span, DUMMY_SP};
use crate::util::nodemap::NodeMap;

/// The DefPathTable maps DefIndexes to DefKeys and vice versa.
/// Internally the DefPathTable holds a tree of DefKeys, where each DefKey
/// stores the DefIndex of its parent.
/// There is one DefPathTable for each crate.
#[derive(Clone, Default, RustcDecodable, RustcEncodable)]
pub struct DefPathTable {
    index_to_key: Vec<DefKey>,
    def_path_hashes: Vec<DefPathHash>,
}

impl DefPathTable {
    fn allocate(&mut self,
                key: DefKey,
                def_path_hash: DefPathHash)
                -> DefIndex {
        let index = {
            let index = DefIndex::from(self.index_to_key.len());
            debug!("DefPathTable::insert() - {:?} <-> {:?}", key, index);
            self.index_to_key.push(key);
            index
        };
        self.def_path_hashes.push(def_path_hash);
        debug_assert!(self.def_path_hashes.len() == self.index_to_key.len());
        index
    }

    pub fn next_id(&self) -> DefIndex {
        DefIndex::from(self.index_to_key.len())
    }

    #[inline(always)]
    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.index_to_key[index.index()].clone()
    }

    #[inline(always)]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        let ret = self.def_path_hashes[index.index()];
        debug!("def_path_hash({:?}) = {:?}", index, ret);
        return ret
    }

    pub fn add_def_path_hashes_to(&self,
                                  cnum: CrateNum,
                                  out: &mut FxHashMap<DefPathHash, DefId>) {
        out.extend(
            self.def_path_hashes
                .iter()
                .enumerate()
                .map(|(index, &hash)| {
                    let def_id = DefId {
                        krate: cnum,
                        index: DefIndex::from(index),
                    };
                    (hash, def_id)
                })
        );
    }

    pub fn size(&self) -> usize {
        self.index_to_key.len()
    }
}

/// The definition table containing node definitions.
/// It holds the `DefPathTable` for local `DefId`s/`DefPath`s and it also stores a
/// mapping from `NodeId`s to local `DefId`s.
#[derive(Clone, Default)]
pub struct Definitions {
    table: DefPathTable,
    node_to_def_index: NodeMap<DefIndex>,
    def_index_to_node: Vec<ast::NodeId>,
    pub(super) node_to_hir_id: IndexVec<ast::NodeId, hir::HirId>,
    /// If `Mark` is an ID of some macro expansion,
    /// then `DefId` is the normal module (`mod`) in which the expanded macro was defined.
    parent_modules_of_macro_defs: FxHashMap<Mark, DefId>,
    /// Item with a given `DefIndex` was defined during macro expansion with ID `Mark`.
    expansions_that_defined: FxHashMap<DefIndex, Mark>,
    next_disambiguator: FxHashMap<(DefIndex, DefPathData), u32>,
    def_index_to_span: FxHashMap<DefIndex, Span>,
}

/// A unique identifier that we can use to lookup a definition
/// precisely. It combines the index of the definition's parent (if
/// any) with a `DisambiguatedDefPathData`.
#[derive(Clone, PartialEq, Debug, Hash, RustcEncodable, RustcDecodable)]
pub struct DefKey {
    /// The parent path.
    pub parent: Option<DefIndex>,

    /// The identifier of this node.
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
        if let Some(name) = data.get_opt_name() {
            name.hash(&mut hasher);
        }

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

/// A pair of `DefPathData` and an integer disambiguator. The integer is
/// normally 0, but in the event that there are multiple defs with the
/// same `parent` and `data`, we use this field to disambiguate
/// between them. This introduces some artificial ordering dependency
/// but means that if you have (e.g.) two impls for the same type in
/// the same module, they do get distinct `DefId`s.
#[derive(Clone, PartialEq, Debug, Hash, RustcEncodable, RustcDecodable)]
pub struct DisambiguatedDefPathData {
    pub data: DefPathData,
    pub disambiguator: u32
}

#[derive(Clone, Debug, Hash, RustcEncodable, RustcDecodable)]
pub struct DefPath {
    /// The path leading from the crate root to the item.
    pub data: Vec<DisambiguatedDefPathData>,

    /// The crate root this path is relative to.
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

    /// Returns a string representation of the `DefPath` without
    /// the crate-prefix. This method is useful if you don't have
    /// a `TyCtxt` available.
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

    /// Returns a filename-friendly string for the `DefPath`, with the
    /// crate-prefix.
    pub fn to_string_friendly<F>(&self, crate_imported_name: F) -> String
        where F: FnOnce(CrateNum) -> Symbol
    {
        let crate_name_str = crate_imported_name(self.krate).as_str();
        let mut s = String::with_capacity(crate_name_str.len() + self.data.len() * 16);

        write!(s, "::{}", crate_name_str).unwrap();

        for component in &self.data {
            if component.disambiguator == 0 {
                write!(s, "::{}", component.data.as_interned_str()).unwrap();
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

    /// Returns a filename-friendly string of the `DefPath`, without
    /// the crate-prefix. This method is useful if you don't have
    /// a `TyCtxt` available.
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum DefPathData {
    // Root: these should only be used for the root nodes, because
    // they are treated specially by the `def_path` function.
    /// The crate root (marker)
    CrateRoot,
    // Catch-all for random DefId things like `DUMMY_NODE_ID`
    Misc,
    // Different kinds of items and item-like things:
    /// An impl
    Impl,
    /// Something in the type NS
    TypeNs(InternedString),
    /// Something in the value NS
    ValueNs(InternedString),
    /// Something in the macro NS
    MacroNs(InternedString),
    /// Something in the lifetime NS
    LifetimeNs(InternedString),
    /// A closure expression
    ClosureExpr,
    // Subportions of items
    /// Implicit ctor for a unit or tuple-like struct or enum variant.
    Ctor,
    /// A constant expression (see {ast,hir}::AnonConst).
    AnonConst,
    /// An `impl Trait` type node
    ImplTrait,
    /// Identifies a piece of crate metadata that is global to a whole crate
    /// (as opposed to just one item). `GlobalMetaData` components are only
    /// supposed to show up right below the crate root.
    GlobalMetaData(InternedString),
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug,
         RustcEncodable, RustcDecodable)]
pub struct DefPathHash(pub Fingerprint);

impl_stable_hash_for!(tuple_struct DefPathHash { fingerprint });

impl Borrow<Fingerprint> for DefPathHash {
    #[inline]
    fn borrow(&self) -> &Fingerprint {
        &self.0
    }
}

impl Definitions {
    pub fn def_path_table(&self) -> &DefPathTable {
        &self.table
    }

    /// Gets the number of definitions.
    pub fn def_index_count(&self) -> usize {
        self.table.index_to_key.len()
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
            let node_id = self.def_index_to_node[def_id.index.index()];
            if node_id != ast::DUMMY_NODE_ID {
                return Some(node_id);
            }
        }
        None
    }

    // FIXME(@ljedrz): replace the NodeId variant
    #[inline]
    pub fn as_local_hir_id(&self, def_id: DefId) -> Option<hir::HirId> {
        if def_id.krate == LOCAL_CRATE {
            let hir_id = self.def_index_to_hir_id(def_id.index);
            if hir_id != hir::DUMMY_HIR_ID {
                Some(hir_id)
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

    #[inline]
    pub fn def_index_to_hir_id(&self, def_index: DefIndex) -> hir::HirId {
        let node_id = self.def_index_to_node[def_index.index()];
        self.node_to_hir_id[node_id]
    }

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate, the span exists
    /// and it's not `DUMMY_SP`.
    #[inline]
    pub fn opt_span(&self, def_id: DefId) -> Option<Span> {
        if def_id.krate == LOCAL_CRATE {
            self.def_index_to_span.get(&def_id.index).cloned()
        } else {
            None
        }
    }

    /// Adds a root definition (no parent) and a few other reserved definitions.
    ///
    /// After the initial definitions are created the first `FIRST_FREE_DEF_INDEX` indexes
    /// are taken, so the "user" indexes will be allocated starting with `FIRST_FREE_DEF_INDEX`
    /// in ascending order.
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
        let root_index = self.table.allocate(key, def_path_hash);
        assert_eq!(root_index, CRATE_DEF_INDEX);
        assert!(self.def_index_to_node.is_empty());
        self.def_index_to_node.push(ast::CRATE_NODE_ID);
        self.node_to_def_index.insert(ast::CRATE_NODE_ID, root_index);

        // Allocate some other DefIndices that always must exist.
        GlobalMetaDataKind::allocate_def_indices(self);

        root_index
    }

    /// Adds a definition with a parent definition.
    pub fn create_def_with_parent(&mut self,
                                  parent: DefIndex,
                                  node_id: ast::NodeId,
                                  data: DefPathData,
                                  expansion: Mark,
                                  span: Span)
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
        let index = self.table.allocate(key, def_path_hash);
        assert_eq!(index.index(), self.def_index_to_node.len());
        self.def_index_to_node.push(node_id);

        // Some things for which we allocate DefIndices don't correspond to
        // anything in the AST, so they don't have a NodeId. For these cases
        // we don't need a mapping from NodeId to DefIndex.
        if node_id != ast::DUMMY_NODE_ID {
            debug!("create_def_with_parent: def_index_to_node[{:?} <-> {:?}", index, node_id);
            self.node_to_def_index.insert(node_id, index);
        }

        if expansion != Mark::root() {
            self.expansions_that_defined.insert(index, expansion);
        }

        // The span is added if it isn't dummy
        if !span.is_dummy() {
            self.def_index_to_span.insert(index, span);
        }

        index
    }

    /// Initialize the `ast::NodeId` to `HirId` mapping once it has been generated during
    /// AST to HIR lowering.
    pub fn init_node_id_to_hir_id_mapping(&mut self,
                                          mapping: IndexVec<ast::NodeId, hir::HirId>) {
        assert!(self.node_to_hir_id.is_empty(),
                "Trying initialize NodeId -> HirId mapping twice");
        self.node_to_hir_id = mapping;
    }

    pub fn expansion_that_defined(&self, index: DefIndex) -> Mark {
        self.expansions_that_defined.get(&index).cloned().unwrap_or(Mark::root())
    }

    pub fn parent_module_of_macro_def(&self, mark: Mark) -> DefId {
        self.parent_modules_of_macro_defs[&mark]
    }

    pub fn add_parent_module_of_macro_def(&mut self, mark: Mark, module: DefId) {
        self.parent_modules_of_macro_defs.insert(mark, module);
    }
}

impl DefPathData {
    pub fn get_opt_name(&self) -> Option<InternedString> {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) |
            ValueNs(name) |
            MacroNs(name) |
            LifetimeNs(name) |
            GlobalMetaData(name) => Some(name),

            Impl |
            CrateRoot |
            Misc |
            ClosureExpr |
            Ctor |
            AnonConst |
            ImplTrait => None
        }
    }

    pub fn as_interned_str(&self) -> InternedString {
        use self::DefPathData::*;
        let s = match *self {
            TypeNs(name) |
            ValueNs(name) |
            MacroNs(name) |
            LifetimeNs(name) |
            GlobalMetaData(name) => {
                return name
            }
            // Note that this does not show up in user print-outs.
            CrateRoot => sym::double_braced_crate,
            Impl => sym::double_braced_impl,
            Misc => sym::double_braced_misc,
            ClosureExpr => sym::double_braced_closure,
            Ctor => sym::double_braced_constructor,
            AnonConst => sym::double_braced_constant,
            ImplTrait => sym::double_braced_opaque,
        };

        s.as_interned_str()
    }

    pub fn to_string(&self) -> String {
        self.as_interned_str().to_string()
    }
}

/// Evaluates to the number of tokens passed to it.
///
/// Logarithmic counting: every one or two recursive expansions, the number of
/// tokens to count is divided by two, instead of being reduced by one.
/// Therefore, the recursion depth is the binary logarithm of the number of
/// tokens to count, and the expanded tree is likewise very small.
macro_rules! count {
    ()                     => (0usize);
    ($one:tt)              => (1usize);
    ($($pairs:tt $_p:tt)*) => (count!($($pairs)*) << 1usize);
    ($odd:tt $($rest:tt)*) => (count!($($rest)*) | 1usize);
}

// We define the GlobalMetaDataKind enum with this macro because we want to
// make sure that we exhaustively iterate over all variants when registering
// the corresponding DefIndices in the DefTable.
macro_rules! define_global_metadata_kind {
    (pub enum GlobalMetaDataKind {
        $($variant:ident),*
    }) => (
        #[derive(Clone, Copy, Debug, Hash, RustcEncodable, RustcDecodable)]
        pub enum GlobalMetaDataKind {
            $($variant),*
        }

        pub const FIRST_FREE_DEF_INDEX: usize = 1 + count!($($variant)*);

        impl GlobalMetaDataKind {
            fn allocate_def_indices(definitions: &mut Definitions) {
                $({
                    let instance = GlobalMetaDataKind::$variant;
                    definitions.create_def_with_parent(
                        CRATE_DEF_INDEX,
                        ast::DUMMY_NODE_ID,
                        DefPathData::GlobalMetaData(instance.name().as_interned_str()),
                        Mark::root(),
                        DUMMY_SP
                    );

                    // Make sure calling def_index does not crash.
                    instance.def_index(&definitions.table);
                })*
            }

            pub fn def_index(&self, def_path_table: &DefPathTable) -> DefIndex {
                let def_key = DefKey {
                    parent: Some(CRATE_DEF_INDEX),
                    disambiguated_data: DisambiguatedDefPathData {
                        data: DefPathData::GlobalMetaData(self.name().as_interned_str()),
                        disambiguator: 0,
                    }
                };

                // These DefKeys are all right after the root,
                // so a linear search is fine.
                let index = def_path_table.index_to_key
                                          .iter()
                                          .position(|k| *k == def_key)
                                          .unwrap();

                DefIndex::from(index)
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
    SourceMap,
    Impls,
    ExportedSymbols
});
