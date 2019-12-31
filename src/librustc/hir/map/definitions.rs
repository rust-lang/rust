//! For each definition, we track the following data. A definition
//! here is defined somewhat circularly as "something with a `DefId`",
//! but it generally corresponds to things like structs, enums, etc.
//! There are also some rather random cases (like const initializer
//! expressions) that are mostly just leftovers.

use crate::hir;
use crate::hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use crate::ich::Fingerprint;
use crate::session::CrateDisambiguator;
use crate::util::nodemap::NodeMap;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_index::vec::IndexVec;
use rustc_span::hygiene::ExpnId;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use std::borrow::Borrow;
use std::fmt::Write;
use std::hash::Hash;
use syntax::ast;

/// The `DefPathTable` maps `DefIndex`es to `DefKey`s and vice versa.
/// Internally the `DefPathTable` holds a tree of `DefKey`s, where each `DefKey`
/// stores the `DefIndex` of its parent.
/// There is one `DefPathTable` for each crate.
#[derive(Clone, Default, RustcDecodable, RustcEncodable)]
pub struct DefPathTable {
    index_to_key: IndexVec<DefIndex, DefKey>,
    def_path_hashes: IndexVec<DefIndex, DefPathHash>,
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
        index
    }

    pub fn next_id(&self) -> DefIndex {
        DefIndex::from(self.index_to_key.len())
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

    pub fn add_def_path_hashes_to(&self, cnum: CrateNum, out: &mut FxHashMap<DefPathHash, DefId>) {
        out.extend(self.def_path_hashes.iter().enumerate().map(|(index, &hash)| {
            let def_id = DefId { krate: cnum, index: DefIndex::from(index) };
            (hash, def_id)
        }));
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
    def_index_to_node: IndexVec<DefIndex, ast::NodeId>,
    pub(super) node_to_hir_id: IndexVec<ast::NodeId, hir::HirId>,
    /// If `ExpnId` is an ID of some macro expansion,
    /// then `DefId` is the normal module (`mod`) in which the expanded macro was defined.
    parent_modules_of_macro_defs: FxHashMap<ExpnId, DefId>,
    /// Item with a given `DefIndex` was defined during macro expansion with ID `ExpnId`.
    expansions_that_defined: FxHashMap<DefIndex, ExpnId>,
    next_disambiguator: FxHashMap<(DefIndex, DefPathData), u32>,
    def_index_to_span: FxHashMap<DefIndex, Span>,
    /// When collecting definitions from an AST fragment produced by a macro invocation `ExpnId`
    /// we know what parent node that fragment should be attached to thanks to this table.
    invocation_parents: FxHashMap<ExpnId, DefIndex>,
    /// Indices of unnamed struct or variant fields with unresolved attributes.
    placeholder_field_indices: NodeMap<usize>,
}

/// A unique identifier that we can use to lookup a definition
/// precisely. It combines the index of the definition's parent (if
/// any) with a `DisambiguatedDefPathData`.
#[derive(Copy, Clone, PartialEq, Debug, RustcEncodable, RustcDecodable)]
pub struct DefKey {
    /// The parent path.
    pub parent: Option<DefIndex>,

    /// The identifier of this node.
    pub disambiguated_data: DisambiguatedDefPathData,
}

impl DefKey {
    fn compute_stable_hash(&self, parent_hash: DefPathHash) -> DefPathHash {
        let mut hasher = StableHasher::new();

        // We hash a `0u8` here to disambiguate between regular `DefPath` hashes,
        // and the special "root_parent" below.
        0u8.hash(&mut hasher);
        parent_hash.hash(&mut hasher);

        let DisambiguatedDefPathData { ref data, disambiguator } = self.disambiguated_data;

        ::std::mem::discriminant(data).hash(&mut hasher);
        if let Some(name) = data.get_opt_name() {
            // Get a stable hash by considering the symbol chars rather than
            // the symbol index.
            name.as_str().hash(&mut hasher);
        }

        disambiguator.hash(&mut hasher);

        DefPathHash(hasher.finish())
    }

    fn root_parent_stable_hash(
        crate_name: &str,
        crate_disambiguator: CrateDisambiguator,
    ) -> DefPathHash {
        let mut hasher = StableHasher::new();
        // Disambiguate this from a regular `DefPath` hash; see `compute_stable_hash()` above.
        1u8.hash(&mut hasher);
        crate_name.hash(&mut hasher);
        crate_disambiguator.hash(&mut hasher);
        DefPathHash(hasher.finish())
    }
}

/// A pair of `DefPathData` and an integer disambiguator. The integer is
/// normally `0`, but in the event that there are multiple defs with the
/// same `parent` and `data`, we use this field to disambiguate
/// between them. This introduces some artificial ordering dependency
/// but means that if you have, e.g., two impls for the same type in
/// the same module, they do get distinct `DefId`s.
#[derive(Copy, Clone, PartialEq, Debug, RustcEncodable, RustcDecodable)]
pub struct DisambiguatedDefPathData {
    pub data: DefPathData,
    pub disambiguator: u32,
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
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
        DefPath { data: data, krate: krate }
    }

    /// Returns a string representation of the `DefPath` without
    /// the crate-prefix. This method is useful if you don't have
    /// a `TyCtxt` available.
    pub fn to_string_no_crate(&self) -> String {
        let mut s = String::with_capacity(self.data.len() * 16);

        for component in &self.data {
            write!(s, "::{}[{}]", component.data.as_symbol(), component.disambiguator).unwrap();
        }

        s
    }

    /// Returns a filename-friendly string for the `DefPath`, with the
    /// crate-prefix.
    pub fn to_string_friendly<F>(&self, crate_imported_name: F) -> String
    where
        F: FnOnce(CrateNum) -> Symbol,
    {
        let crate_name_str = crate_imported_name(self.krate).as_str();
        let mut s = String::with_capacity(crate_name_str.len() + self.data.len() * 16);

        write!(s, "::{}", crate_name_str).unwrap();

        for component in &self.data {
            if component.disambiguator == 0 {
                write!(s, "::{}", component.data.as_symbol()).unwrap();
            } else {
                write!(s, "{}[{}]", component.data.as_symbol(), component.disambiguator).unwrap();
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
                write!(s, "{}", component.data.as_symbol()).unwrap();
            } else {
                write!(s, "{}[{}]", component.data.as_symbol(), component.disambiguator).unwrap();
            }
        }
        s
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
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

#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    RustcEncodable,
    RustcDecodable,
    HashStable
)]
pub struct DefPathHash(pub Fingerprint);

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
        self.node_to_def_index.get(&node).copied()
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
            let node_id = self.def_index_to_node[def_id.index];
            if node_id != ast::DUMMY_NODE_ID {
                return Some(node_id);
            }
        }
        None
    }

    #[inline]
    pub fn as_local_hir_id(&self, def_id: DefId) -> Option<hir::HirId> {
        if def_id.krate == LOCAL_CRATE {
            let hir_id = self.def_index_to_hir_id(def_id.index);
            if hir_id != hir::DUMMY_HIR_ID { Some(hir_id) } else { None }
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
        let node_id = self.def_index_to_node[def_index];
        self.node_to_hir_id[node_id]
    }

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate, the span exists
    /// and it's not `DUMMY_SP`.
    #[inline]
    pub fn opt_span(&self, def_id: DefId) -> Option<Span> {
        if def_id.krate == LOCAL_CRATE {
            self.def_index_to_span.get(&def_id.index).copied()
        } else {
            None
        }
    }

    /// Adds a root definition (no parent) and a few other reserved definitions.
    pub fn create_root_def(
        &mut self,
        crate_name: &str,
        crate_disambiguator: CrateDisambiguator,
    ) -> DefIndex {
        let key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0,
            },
        };

        let parent_hash = DefKey::root_parent_stable_hash(crate_name, crate_disambiguator);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        // Create the definition.
        let root_index = self.table.allocate(key, def_path_hash);
        assert_eq!(root_index, CRATE_DEF_INDEX);
        assert!(self.def_index_to_node.is_empty());
        self.def_index_to_node.push(ast::CRATE_NODE_ID);
        self.node_to_def_index.insert(ast::CRATE_NODE_ID, root_index);
        self.set_invocation_parent(ExpnId::root(), root_index);

        root_index
    }

    /// Adds a definition with a parent definition.
    pub fn create_def_with_parent(
        &mut self,
        parent: DefIndex,
        node_id: ast::NodeId,
        data: DefPathData,
        expn_id: ExpnId,
        span: Span,
    ) -> DefIndex {
        debug!(
            "create_def_with_parent(parent={:?}, node_id={:?}, data={:?})",
            parent, node_id, data
        );

        assert!(
            !self.node_to_def_index.contains_key(&node_id),
            "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
            node_id,
            data,
            self.table.def_key(self.node_to_def_index[&node_id])
        );

        // The root node must be created with `create_root_def()`.
        assert!(data != DefPathData::CrateRoot);

        // Find the next free disambiguator for this key.
        let disambiguator = {
            let next_disamb = self.next_disambiguator.entry((parent, data)).or_insert(0);
            let disambiguator = *next_disamb;
            *next_disamb = next_disamb.checked_add(1).expect("disambiguator overflow");
            disambiguator
        };

        let key = DefKey {
            parent: Some(parent),
            disambiguated_data: DisambiguatedDefPathData { data, disambiguator },
        };

        let parent_hash = self.table.def_path_hash(parent);
        let def_path_hash = key.compute_stable_hash(parent_hash);

        debug!("create_def_with_parent: after disambiguation, key = {:?}", key);

        // Create the definition.
        let index = self.table.allocate(key, def_path_hash);
        assert_eq!(index.index(), self.def_index_to_node.len());
        self.def_index_to_node.push(node_id);

        // Some things for which we allocate `DefIndex`es don't correspond to
        // anything in the AST, so they don't have a `NodeId`. For these cases
        // we don't need a mapping from `NodeId` to `DefIndex`.
        if node_id != ast::DUMMY_NODE_ID {
            debug!("create_def_with_parent: def_index_to_node[{:?} <-> {:?}", index, node_id);
            self.node_to_def_index.insert(node_id, index);
        }

        if expn_id != ExpnId::root() {
            self.expansions_that_defined.insert(index, expn_id);
        }

        // The span is added if it isn't dummy.
        if !span.is_dummy() {
            self.def_index_to_span.insert(index, span);
        }

        index
    }

    /// Initializes the `ast::NodeId` to `HirId` mapping once it has been generated during
    /// AST to HIR lowering.
    pub fn init_node_id_to_hir_id_mapping(&mut self, mapping: IndexVec<ast::NodeId, hir::HirId>) {
        assert!(
            self.node_to_hir_id.is_empty(),
            "trying to initialize `NodeId` -> `HirId` mapping twice"
        );
        self.node_to_hir_id = mapping;
    }

    pub fn expansion_that_defined(&self, index: DefIndex) -> ExpnId {
        self.expansions_that_defined.get(&index).copied().unwrap_or(ExpnId::root())
    }

    pub fn parent_module_of_macro_def(&self, expn_id: ExpnId) -> DefId {
        self.parent_modules_of_macro_defs[&expn_id]
    }

    pub fn add_parent_module_of_macro_def(&mut self, expn_id: ExpnId, module: DefId) {
        self.parent_modules_of_macro_defs.insert(expn_id, module);
    }

    pub fn invocation_parent(&self, invoc_id: ExpnId) -> DefIndex {
        self.invocation_parents[&invoc_id]
    }

    pub fn set_invocation_parent(&mut self, invoc_id: ExpnId, parent: DefIndex) {
        let old_parent = self.invocation_parents.insert(invoc_id, parent);
        assert!(old_parent.is_none(), "parent `DefIndex` is reset for an invocation");
    }

    pub fn placeholder_field_index(&self, node_id: ast::NodeId) -> usize {
        self.placeholder_field_indices[&node_id]
    }

    pub fn set_placeholder_field_index(&mut self, node_id: ast::NodeId, index: usize) {
        let old_index = self.placeholder_field_indices.insert(node_id, index);
        assert!(old_index.is_none(), "placeholder field index is reset for a node ID");
    }
}

impl DefPathData {
    pub fn get_opt_name(&self) -> Option<Symbol> {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name) => Some(name),

            Impl | CrateRoot | Misc | ClosureExpr | Ctor | AnonConst | ImplTrait => None,
        }
    }

    pub fn as_symbol(&self) -> Symbol {
        use self::DefPathData::*;
        match *self {
            TypeNs(name) | ValueNs(name) | MacroNs(name) | LifetimeNs(name) => name,
            // Note that this does not show up in user print-outs.
            CrateRoot => sym::double_braced_crate,
            Impl => sym::double_braced_impl,
            Misc => sym::double_braced_misc,
            ClosureExpr => sym::double_braced_closure,
            Ctor => sym::double_braced_constructor,
            AnonConst => sym::double_braced_constant,
            ImplTrait => sym::double_braced_opaque,
        }
    }

    pub fn to_string(&self) -> String {
        self.as_symbol().to_string()
    }
}
