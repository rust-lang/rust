use self::collector::NodeCollector;
pub use self::def_collector::{DefCollector, MacroInvocationData};
pub use self::definitions::{
    Definitions, DefKey, DefPath, DefPathData, DisambiguatedDefPathData, DefPathHash
};

use crate::dep_graph::{DepGraph, DepNode, DepKind, DepNodeIndex};

use crate::hir::def_id::{CRATE_DEF_INDEX, DefId, LocalDefId};

use crate::middle::cstore::CrateStoreDyn;

use rustc_target::spec::abi::Abi;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::indexed_vec::IndexVec;
use syntax::ast::{self, Name, NodeId};
use syntax::source_map::Spanned;
use syntax::ext::base::MacroKind;
use syntax_pos::{Span, DUMMY_SP};

use crate::hir::*;
use crate::hir::DefKind;
use crate::hir::itemlikevisit::ItemLikeVisitor;
use crate::hir::print::Nested;
use crate::util::nodemap::FxHashMap;
use crate::util::common::time;

use std::result::Result::Err;
use crate::ty::query::Providers;

pub mod blocks;
mod collector;
mod def_collector;
pub mod definitions;
mod hir_id_validator;

/// Represents an entry and its parent `HirId`.
#[derive(Copy, Clone, Debug)]
pub struct Entry<'hir> {
    parent: HirId,
    dep_node: DepNodeIndex,
    node: Node<'hir>,
}

impl<'hir> Entry<'hir> {
    fn parent_node(self) -> Option<HirId> {
        match self.node {
            Node::Crate | Node::MacroDef(_) => None,
            _ => Some(self.parent),
        }
    }

    fn fn_decl(&self) -> Option<&'hir FnDecl> {
        match self.node {
            Node::Item(ref item) => {
                match item.node {
                    ItemKind::Fn(ref fn_decl, _, _, _) => Some(fn_decl),
                    _ => None,
                }
            }

            Node::TraitItem(ref item) => {
                match item.node {
                    TraitItemKind::Method(ref method_sig, _) => Some(&method_sig.decl),
                    _ => None
                }
            }

            Node::ImplItem(ref item) => {
                match item.node {
                    ImplItemKind::Method(ref method_sig, _) => Some(&method_sig.decl),
                    _ => None,
                }
            }

            Node::Expr(ref expr) => {
                match expr.node {
                    ExprKind::Closure(_, ref fn_decl, ..) => Some(fn_decl),
                    _ => None,
                }
            }

            _ => None,
        }
    }

    fn associated_body(self) -> Option<BodyId> {
        match self.node {
            Node::Item(item) => {
                match item.node {
                    ItemKind::Const(_, body) |
                    ItemKind::Static(.., body) |
                    ItemKind::Fn(_, _, _, body) => Some(body),
                    _ => None,
                }
            }

            Node::TraitItem(item) => {
                match item.node {
                    TraitItemKind::Const(_, Some(body)) |
                    TraitItemKind::Method(_, TraitMethod::Provided(body)) => Some(body),
                    _ => None
                }
            }

            Node::ImplItem(item) => {
                match item.node {
                    ImplItemKind::Const(_, body) |
                    ImplItemKind::Method(_, body) => Some(body),
                    _ => None,
                }
            }

            Node::AnonConst(constant) => Some(constant.body),

            Node::Expr(expr) => {
                match expr.node {
                    ExprKind::Closure(.., body, _, _) => Some(body),
                    _ => None,
                }
            }

            _ => None
        }
    }

    fn is_body_owner(self, hir_id: HirId) -> bool {
        match self.associated_body() {
            Some(b) => b.hir_id == hir_id,
            None => false,
        }
    }
}

/// Stores a crate and any number of inlined items from other crates.
pub struct Forest {
    krate: Crate,
    pub dep_graph: DepGraph,
}

impl Forest {
    pub fn new(krate: Crate, dep_graph: &DepGraph) -> Forest {
        Forest {
            krate,
            dep_graph: dep_graph.clone(),
        }
    }

    pub fn krate(&self) -> &Crate {
        self.dep_graph.read(DepNode::new_no_params(DepKind::Krate));
        &self.krate
    }

    /// This is used internally in the dependency tracking system.
    /// Use the `krate` method to ensure your dependency on the
    /// crate is tracked.
    pub fn untracked_krate(&self) -> &Crate {
        &self.krate
    }
}

/// This type is effectively a `HashMap<HirId, Entry<'hir>>`,
/// but it is implemented as 2 layers of arrays.
/// - first we have `A = Vec<Option<B>>` mapping a `DefIndex`'s index to an inner value
/// - which is `B = IndexVec<ItemLocalId, Option<Entry<'hir>>` which gives you the `Entry`.
pub(super) type HirEntryMap<'hir> = Vec<Option<IndexVec<ItemLocalId, Option<Entry<'hir>>>>>;

/// Represents a mapping from `NodeId`s to AST elements and their parent `NodeId`s.
#[derive(Clone)]
pub struct Map<'hir> {
    /// The backing storage for all the AST nodes.
    pub forest: &'hir Forest,

    /// Same as the dep_graph in forest, just available with one fewer
    /// deref. This is a gratuitous micro-optimization.
    pub dep_graph: DepGraph,

    /// The SVH of the local crate.
    pub crate_hash: Svh,

    map: HirEntryMap<'hir>,

    definitions: &'hir Definitions,

    /// The reverse mapping of `node_to_hir_id`.
    hir_to_node_id: FxHashMap<HirId, NodeId>,
}

impl<'hir> Map<'hir> {
    #[inline]
    fn lookup(&self, id: HirId) -> Option<&Entry<'hir>> {
        let local_map = self.map.get(id.owner.index())?;
        local_map.as_ref()?.get(id.local_id)?.as_ref()
    }

    /// Registers a read in the dependency graph of the AST node with
    /// the given `id`. This needs to be called each time a public
    /// function returns the HIR for a node -- in other words, when it
    /// "reveals" the content of a node to the caller (who might not
    /// otherwise have had access to those contents, and hence needs a
    /// read recorded). If the function just returns a DefId or
    /// HirId, no actual content was returned, so no read is needed.
    pub fn read(&self, hir_id: HirId) {
        if let Some(entry) = self.lookup(hir_id) {
            self.dep_graph.read_index(entry.dep_node);
        } else {
            bug!("called `HirMap::read()` with invalid `HirId`: {:?}", hir_id)
        }
    }

    #[inline]
    pub fn definitions(&self) -> &'hir Definitions {
        self.definitions
    }

    pub fn def_key(&self, def_id: DefId) -> DefKey {
        assert!(def_id.is_local());
        self.definitions.def_key(def_id.index)
    }

    pub fn def_path_from_hir_id(&self, id: HirId) -> Option<DefPath> {
        self.opt_local_def_id_from_hir_id(id).map(|def_id| {
            self.def_path(def_id)
        })
    }

    pub fn def_path(&self, def_id: DefId) -> DefPath {
        assert!(def_id.is_local());
        self.definitions.def_path(def_id.index)
    }

    #[inline]
    pub fn local_def_id(&self, node: NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap_or_else(|| {
            let hir_id = self.node_to_hir_id(node);
            bug!("local_def_id: no entry for `{}`, which has a map of `{:?}`",
                 node, self.find_entry(hir_id))
        })
    }

    // FIXME(@ljedrz): replace the `NodeId` variant.
    #[inline]
    pub fn local_def_id_from_hir_id(&self, hir_id: HirId) -> DefId {
        self.opt_local_def_id_from_hir_id(hir_id).unwrap_or_else(|| {
            bug!("local_def_id_from_hir_id: no entry for `{:?}`, which has a map of `{:?}`",
                 hir_id, self.find_entry(hir_id))
        })
    }

    // FIXME(@ljedrz): replace the `NodeId` variant.
    #[inline]
    pub fn opt_local_def_id_from_hir_id(&self, hir_id: HirId) -> Option<DefId> {
        let node_id = self.hir_to_node_id(hir_id);
        self.definitions.opt_local_def_id(node_id)
    }

    #[inline]
    pub fn opt_local_def_id(&self, node: NodeId) -> Option<DefId> {
        self.definitions.opt_local_def_id(node)
    }

    #[inline]
    pub fn as_local_node_id(&self, def_id: DefId) -> Option<NodeId> {
        self.definitions.as_local_node_id(def_id)
    }

    // FIXME(@ljedrz): replace the `NodeId` variant.
    #[inline]
    pub fn as_local_hir_id(&self, def_id: DefId) -> Option<HirId> {
        self.definitions.as_local_hir_id(def_id)
    }

    #[inline]
    pub fn hir_to_node_id(&self, hir_id: HirId) -> NodeId {
        self.hir_to_node_id[&hir_id]
    }

    #[inline]
    pub fn node_to_hir_id(&self, node_id: NodeId) -> HirId {
        self.definitions.node_to_hir_id(node_id)
    }

    #[inline]
    pub fn def_index_to_hir_id(&self, def_index: DefIndex) -> HirId {
        self.definitions.def_index_to_hir_id(def_index)
    }

    #[inline]
    pub fn local_def_id_to_hir_id(&self, def_id: LocalDefId) -> HirId {
        self.definitions.def_index_to_hir_id(def_id.to_def_id().index)
    }

    fn def_kind(&self, hir_id: HirId) -> Option<DefKind> {
        let node = if let Some(node) = self.find(hir_id) {
            node
        } else {
            return None
        };

        Some(match node {
            Node::Item(item) => {
                match item.node {
                    ItemKind::Static(..) => DefKind::Static,
                    ItemKind::Const(..) => DefKind::Const,
                    ItemKind::Fn(..) => DefKind::Fn,
                    ItemKind::Mod(..) => DefKind::Mod,
                    ItemKind::Existential(..) => DefKind::Existential,
                    ItemKind::Ty(..) => DefKind::TyAlias,
                    ItemKind::Enum(..) => DefKind::Enum,
                    ItemKind::Struct(..) => DefKind::Struct,
                    ItemKind::Union(..) => DefKind::Union,
                    ItemKind::Trait(..) => DefKind::Trait,
                    ItemKind::TraitAlias(..) => DefKind::TraitAlias,
                    ItemKind::ExternCrate(_) |
                    ItemKind::Use(..) |
                    ItemKind::ForeignMod(..) |
                    ItemKind::GlobalAsm(..) |
                    ItemKind::Impl(..) => return None,
                }
            }
            Node::ForeignItem(item) => {
                match item.node {
                    ForeignItemKind::Fn(..) => DefKind::Fn,
                    ForeignItemKind::Static(..) => DefKind::Static,
                    ForeignItemKind::Type => DefKind::ForeignTy,
                }
            }
            Node::TraitItem(item) => {
                match item.node {
                    TraitItemKind::Const(..) => DefKind::AssocConst,
                    TraitItemKind::Method(..) => DefKind::Method,
                    TraitItemKind::Type(..) => DefKind::AssocTy,
                }
            }
            Node::ImplItem(item) => {
                match item.node {
                    ImplItemKind::Const(..) => DefKind::AssocConst,
                    ImplItemKind::Method(..) => DefKind::Method,
                    ImplItemKind::Type(..) => DefKind::AssocTy,
                    ImplItemKind::Existential(..) => DefKind::AssocExistential,
                }
            }
            Node::Variant(_) => DefKind::Variant,
            Node::Ctor(variant_data) => {
                // FIXME(eddyb) is this even possible, if we have a `Node::Ctor`?
                if variant_data.ctor_hir_id().is_none() {
                    return None;
                }
                let ctor_of = match self.find(self.get_parent_node(hir_id)) {
                    Some(Node::Item(..)) => def::CtorOf::Struct,
                    Some(Node::Variant(..)) => def::CtorOf::Variant,
                    _ => unreachable!(),
                };
                DefKind::Ctor(ctor_of, def::CtorKind::from_hir(variant_data))
            }
            Node::AnonConst(_) |
            Node::Field(_) |
            Node::Expr(_) |
            Node::Stmt(_) |
            Node::PathSegment(_) |
            Node::Ty(_) |
            Node::TraitRef(_) |
            Node::Pat(_) |
            Node::Binding(_) |
            Node::Local(_) |
            Node::Arm(_) |
            Node::Lifetime(_) |
            Node::Visibility(_) |
            Node::Block(_) |
            Node::Crate => return None,
            Node::MacroDef(_) => DefKind::Macro(MacroKind::Bang),
            Node::GenericParam(param) => {
                match param.kind {
                    GenericParamKind::Lifetime { .. } => return None,
                    GenericParamKind::Type { .. } => DefKind::TyParam,
                    GenericParamKind::Const { .. } => DefKind::ConstParam,
                }
            }
        })
    }

    fn find_entry(&self, id: HirId) -> Option<Entry<'hir>> {
        self.lookup(id).cloned()
    }

    pub fn krate(&self) -> &'hir Crate {
        self.forest.krate()
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_item(id)
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.impl_item(id)
    }

    pub fn body(&self, id: BodyId) -> &'hir Body {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.body(id)
    }

    pub fn fn_decl_by_hir_id(&self, hir_id: HirId) -> Option<&'hir FnDecl> {
        if let Some(entry) = self.find_entry(hir_id) {
            entry.fn_decl()
        } else {
            bug!("no entry for hir_id `{}`", hir_id)
        }
    }

    /// Returns the `HirId` that corresponds to the definition of
    /// which this is the body of, i.e., a `fn`, `const` or `static`
    /// item (possibly associated), a closure, or a `hir::AnonConst`.
    pub fn body_owner(&self, BodyId { hir_id }: BodyId) -> HirId {
        let parent = self.get_parent_node(hir_id);
        assert!(self.lookup(parent).map_or(false, |e| e.is_body_owner(hir_id)));
        parent
    }

    pub fn body_owner_def_id(&self, id: BodyId) -> DefId {
        self.local_def_id_from_hir_id(self.body_owner(id))
    }

    /// Given a `HirId`, returns the `BodyId` associated with it,
    /// if the node is a body owner, otherwise returns `None`.
    pub fn maybe_body_owned_by(&self, hir_id: HirId) -> Option<BodyId> {
        if let Some(entry) = self.find_entry(hir_id) {
            if self.dep_graph.is_fully_enabled() {
                let hir_id_owner = hir_id.owner;
                let def_path_hash = self.definitions.def_path_hash(hir_id_owner);
                self.dep_graph.read(def_path_hash.to_dep_node(DepKind::HirBody));
            }

            entry.associated_body()
        } else {
            bug!("no entry for id `{}`", hir_id)
        }
    }

    /// Given a body owner's id, returns the `BodyId` associated with it.
    pub fn body_owned_by(&self, id: HirId) -> BodyId {
        self.maybe_body_owned_by(id).unwrap_or_else(|| {
            span_bug!(self.span(id), "body_owned_by: {} has no associated body",
                      self.node_to_string(id));
        })
    }

    pub fn body_owner_kind(&self, id: HirId) -> BodyOwnerKind {
        match self.get(id) {
            Node::Item(&Item { node: ItemKind::Const(..), .. }) |
            Node::TraitItem(&TraitItem { node: TraitItemKind::Const(..), .. }) |
            Node::ImplItem(&ImplItem { node: ImplItemKind::Const(..), .. }) |
            Node::AnonConst(_) => {
                BodyOwnerKind::Const
            }
            Node::Ctor(..) |
            Node::Item(&Item { node: ItemKind::Fn(..), .. }) |
            Node::TraitItem(&TraitItem { node: TraitItemKind::Method(..), .. }) |
            Node::ImplItem(&ImplItem { node: ImplItemKind::Method(..), .. }) => {
                BodyOwnerKind::Fn
            }
            Node::Item(&Item { node: ItemKind::Static(_, m, _), .. }) => {
                BodyOwnerKind::Static(m)
            }
            Node::Expr(&Expr { node: ExprKind::Closure(..), .. }) => {
                BodyOwnerKind::Closure
            }
            node => bug!("{:#?} is not a body node", node),
        }
    }

    pub fn ty_param_owner(&self, id: HirId) -> HirId {
        match self.get(id) {
            Node::Item(&Item { node: ItemKind::Trait(..), .. }) |
            Node::Item(&Item { node: ItemKind::TraitAlias(..), .. }) => id,
            Node::GenericParam(_) => self.get_parent_node(id),
            _ => bug!("ty_param_owner: {} not a type parameter", self.node_to_string(id))
        }
    }

    pub fn ty_param_name(&self, id: HirId) -> Name {
        match self.get(id) {
            Node::Item(&Item { node: ItemKind::Trait(..), .. }) |
            Node::Item(&Item { node: ItemKind::TraitAlias(..), .. }) => kw::SelfUpper,
            Node::GenericParam(param) => param.name.ident().name,
            _ => bug!("ty_param_name: {} not a type parameter", self.node_to_string(id)),
        }
    }

    pub fn trait_impls(&self, trait_did: DefId) -> &'hir [HirId] {
        self.dep_graph.read(DepNode::new_no_params(DepKind::AllLocalTraitImpls));

        // N.B., intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_impls.get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    /// Gets the attributes on the crate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'hir [ast::Attribute] {
        let def_path_hash = self.definitions.def_path_hash(CRATE_DEF_INDEX);

        self.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
        &self.forest.krate.attrs
    }

    pub fn get_module(&self, module: DefId) -> (&'hir Mod, Span, HirId)
    {
        let hir_id = self.as_local_hir_id(module).unwrap();
        self.read(hir_id);
        match self.find_entry(hir_id).unwrap().node {
            Node::Item(&Item {
                span,
                node: ItemKind::Mod(ref m),
                ..
            }) => (m, span, hir_id),
            Node::Crate => (&self.forest.krate.module, self.forest.krate.span, hir_id),
            _ => panic!("not a module")
        }
    }

    pub fn visit_item_likes_in_module<V>(&self, module: DefId, visitor: &mut V)
        where V: ItemLikeVisitor<'hir>
    {
        let hir_id = self.as_local_hir_id(module).unwrap();

        // Read the module so we'll be re-executed if new items
        // appear immediately under in the module. If some new item appears
        // in some nested item in the module, we'll be re-executed due to reads
        // in the expect_* calls the loops below
        self.read(hir_id);

        let node_id = self.hir_to_node_id[&hir_id];

        let module = &self.forest.krate.modules[&node_id];

        for id in &module.items {
            visitor.visit_item(self.expect_item(*id));
        }

        for id in &module.trait_items {
            visitor.visit_trait_item(self.expect_trait_item(id.hir_id));
        }

        for id in &module.impl_items {
            visitor.visit_impl_item(self.expect_impl_item(id.hir_id));
        }
    }

    /// Retrieves the `Node` corresponding to `id`, panicking if it cannot be found.
    pub fn get(&self, id: HirId) -> Node<'hir> {
        // read recorded by `find`
        self.find(id).unwrap_or_else(||
            bug!("couldn't find hir id {} in the HIR map", id))
    }

    pub fn get_if_local(&self, id: DefId) -> Option<Node<'hir>> {
        self.as_local_hir_id(id).map(|id| self.get(id)) // read recorded by `get`
    }

    pub fn get_generics(&self, id: DefId) -> Option<&'hir Generics> {
        self.get_if_local(id).and_then(|node| {
            match node {
                Node::ImplItem(ref impl_item) => Some(&impl_item.generics),
                Node::TraitItem(ref trait_item) => Some(&trait_item.generics),
                Node::Item(ref item) => {
                    match item.node {
                        ItemKind::Fn(_, _, ref generics, _) |
                        ItemKind::Ty(_, ref generics) |
                        ItemKind::Enum(_, ref generics) |
                        ItemKind::Struct(_, ref generics) |
                        ItemKind::Union(_, ref generics) |
                        ItemKind::Trait(_, _, ref generics, ..) |
                        ItemKind::TraitAlias(ref generics, _) |
                        ItemKind::Impl(_, _, _, ref generics, ..) => Some(generics),
                        _ => None,
                    }
                }
                _ => None,
            }
        })
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    pub fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        let result = self.find_entry(hir_id).and_then(|entry| {
            if let Node::Crate = entry.node {
                None
            } else {
                Some(entry.node)
            }
        });
        if result.is_some() {
            self.read(hir_id);
        }
        result
    }

    /// Similar to `get_parent`; returns the parent HIR Id, or just `hir_id` if there
    /// is no parent. Note that the parent may be `CRATE_HIR_ID`, which is not itself
    /// present in the map, so passing the return value of `get_parent_node` to
    /// `get` may in fact panic.
    /// This function returns the immediate parent in the HIR, whereas `get_parent`
    /// returns the enclosing item. Note that this might not be the actual parent
    /// node in the HIR -- some kinds of nodes are not in the map and these will
    /// never appear as the parent node. Thus, you can always walk the parent nodes
    /// from a node to the root of the HIR (unless you get back the same ID here,
    /// which can happen if the ID is not in the map itself or is just weird).
    pub fn get_parent_node(&self, hir_id: HirId) -> HirId {
        if self.dep_graph.is_fully_enabled() {
            let hir_id_owner = hir_id.owner;
            let def_path_hash = self.definitions.def_path_hash(hir_id_owner);
            self.dep_graph.read(def_path_hash.to_dep_node(DepKind::HirBody));
        }

        self.find_entry(hir_id)
            .and_then(|x| x.parent_node())
            .unwrap_or(hir_id)
    }

    /// Check if the node is an argument. An argument is a local variable whose
    /// immediate parent is an item or a closure.
    pub fn is_argument(&self, id: HirId) -> bool {
        match self.find(id) {
            Some(Node::Binding(_)) => (),
            _ => return false,
        }
        match self.find(self.get_parent_node(id)) {
            Some(Node::Item(_)) |
            Some(Node::TraitItem(_)) |
            Some(Node::ImplItem(_)) => true,
            Some(Node::Expr(e)) => {
                match e.node {
                    ExprKind::Closure(..) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn is_const_scope(&self, hir_id: HirId) -> bool {
        self.walk_parent_nodes(hir_id, |node| match *node {
            Node::Item(Item { node: ItemKind::Const(_, _), .. }) => true,
            Node::Item(Item { node: ItemKind::Fn(_, header, _, _), .. }) => header.is_const(),
            _ => false,
        }, |_| false).map(|id| id != CRATE_HIR_ID).unwrap_or(false)
    }

    /// If there is some error when walking the parents (e.g., a node does not
    /// have a parent in the map or a node can't be found), then we return the
    /// last good `HirId` we found. Note that reaching the crate root (`id == 0`),
    /// is not an error, since items in the crate module have the crate root as
    /// parent.
    fn walk_parent_nodes<F, F2>(&self,
                                start_id: HirId,
                                found: F,
                                bail_early: F2)
        -> Result<HirId, HirId>
        where F: Fn(&Node<'hir>) -> bool, F2: Fn(&Node<'hir>) -> bool
    {
        let mut id = start_id;
        loop {
            let parent_id = self.get_parent_node(id);
            if parent_id == CRATE_HIR_ID {
                return Ok(CRATE_HIR_ID);
            }
            if parent_id == id {
                return Err(id);
            }

            if let Some(entry) = self.find_entry(parent_id) {
                if let Node::Crate = entry.node {
                    return Err(id);
                }
                if found(&entry.node) {
                    return Ok(parent_id);
                } else if bail_early(&entry.node) {
                    return Err(parent_id);
                }
                id = parent_id;
            } else {
                return Err(id);
            }
        }
    }

    /// Retrieves the `HirId` for `id`'s enclosing method, unless there's a
    /// `while` or `loop` before reaching it, as block tail returns are not
    /// available in them.
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     if x == 1 {
    ///         true  // `get_return_block` gets passed the `id` corresponding
    ///     } else {  // to this, it will return `foo`'s `HirId`.
    ///         false
    ///     }
    /// }
    /// ```
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     loop {
    ///         true  // `get_return_block` gets passed the `id` corresponding
    ///     }         // to this, it will return `None`.
    ///     false
    /// }
    /// ```
    pub fn get_return_block(&self, id: HirId) -> Option<HirId> {
        let match_fn = |node: &Node<'_>| {
            match *node {
                Node::Item(_) |
                Node::ForeignItem(_) |
                Node::TraitItem(_) |
                Node::Expr(Expr { node: ExprKind::Closure(..), ..}) |
                Node::ImplItem(_) => true,
                _ => false,
            }
        };
        let match_non_returning_block = |node: &Node<'_>| {
            match *node {
                Node::Expr(ref expr) => {
                    match expr.node {
                        ExprKind::While(..) | ExprKind::Loop(..) | ExprKind::Ret(..) => true,
                        _ => false,
                    }
                }
                _ => false,
            }
        };

        self.walk_parent_nodes(id, match_fn, match_non_returning_block).ok()
    }

    /// Retrieves the `HirId` for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the HIR which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent_item(&self, hir_id: HirId) -> HirId {
        match self.walk_parent_nodes(hir_id, |node| match *node {
            Node::Item(_) |
            Node::ForeignItem(_) |
            Node::TraitItem(_) |
            Node::ImplItem(_) => true,
            _ => false,
        }, |_| false) {
            Ok(id) => id,
            Err(id) => id,
        }
    }

    /// Returns the `DefId` of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub fn get_module_parent(&self, id: HirId) -> DefId {
        self.local_def_id_from_hir_id(self.get_module_parent_node(id))
    }

    /// Returns the `HirId` of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub fn get_module_parent_node(&self, hir_id: HirId) -> HirId {
        match self.walk_parent_nodes(hir_id, |node| match *node {
            Node::Item(&Item { node: ItemKind::Mod(_), .. }) => true,
            _ => false,
        }, |_| false) {
            Ok(id) => id,
            Err(id) => id,
        }
    }

    /// Returns the nearest enclosing scope. A scope is roughly an item or block.
    pub fn get_enclosing_scope(&self, hir_id: HirId) -> Option<HirId> {
        self.walk_parent_nodes(hir_id, |node| match *node {
            Node::Item(i) => {
                match i.node {
                    ItemKind::Fn(..)
                    | ItemKind::Mod(..)
                    | ItemKind::Enum(..)
                    | ItemKind::Struct(..)
                    | ItemKind::Union(..)
                    | ItemKind::Trait(..)
                    | ItemKind::Impl(..) => true,
                    _ => false,
                }
            },
            Node::ForeignItem(fi) => {
                match fi.node {
                    ForeignItemKind::Fn(..) => true,
                    _ => false,
                }
            },
            Node::TraitItem(ti) => {
                match ti.node {
                    TraitItemKind::Method(..) => true,
                    _ => false,
                }
            },
            Node::ImplItem(ii) => {
                match ii.node {
                    ImplItemKind::Method(..) => true,
                    _ => false,
                }
            },
            Node::Block(_) => true,
            _ => false,
        }, |_| false).ok()
    }

    /// Returns the defining scope for an existential type definition.
    pub fn get_defining_scope(&self, id: HirId) -> Option<HirId> {
        let mut scope = id;
        loop {
            scope = self.get_enclosing_scope(scope)?;
            if scope == CRATE_HIR_ID {
                return Some(CRATE_HIR_ID);
            }
            match self.get(scope) {
                Node::Item(i) => {
                    match i.node {
                        ItemKind::Existential(ExistTy { impl_trait_fn: None, .. }) => {}
                        _ => break,
                    }
                }
                Node::Block(_) => {}
                _ => break,
            }
        }
        Some(scope)
    }

    pub fn get_parent_did(&self, id: HirId) -> DefId {
        self.local_def_id_from_hir_id(self.get_parent_item(id))
    }

    pub fn get_foreign_abi(&self, hir_id: HirId) -> Abi {
        let parent = self.get_parent_item(hir_id);
        if let Some(entry) = self.find_entry(parent) {
            if let Entry {
                node: Node::Item(Item { node: ItemKind::ForeignMod(ref nm), .. }), .. } = entry
            {
                self.read(hir_id); // reveals some of the content of a node
                return nm.abi;
            }
        }
        bug!("expected foreign mod or inlined parent, found {}", self.node_to_string(parent))
    }

    pub fn expect_item(&self, id: HirId) -> &'hir Item {
        match self.find(id) { // read recorded by `find`
            Some(Node::Item(item)) => item,
            _ => bug!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_impl_item(&self, id: HirId) -> &'hir ImplItem {
        match self.find(id) {
            Some(Node::ImplItem(item)) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_trait_item(&self, id: HirId) -> &'hir TraitItem {
        match self.find(id) {
            Some(Node::TraitItem(item)) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_variant_data(&self, id: HirId) -> &'hir VariantData {
        match self.find(id) {
            Some(Node::Item(i)) => {
                match i.node {
                    ItemKind::Struct(ref struct_def, _) |
                    ItemKind::Union(ref struct_def, _) => struct_def,
                    _ => bug!("struct ID bound to non-struct {}", self.node_to_string(id))
                }
            }
            Some(Node::Variant(variant)) => &variant.node.data,
            Some(Node::Ctor(data)) => data,
            _ => bug!("expected struct or variant, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_variant(&self, id: HirId) -> &'hir Variant {
        match self.find(id) {
            Some(Node::Variant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(&self, id: HirId) -> &'hir ForeignItem {
        match self.find(id) {
            Some(Node::ForeignItem(item)) => item,
            _ => bug!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_expr(&self, id: HirId) -> &'hir Expr {
        match self.find(id) { // read recorded by find
            Some(Node::Expr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id))
        }
    }

    pub fn name(&self, id: HirId) -> Name {
        match self.get(id) {
            Node::Item(i) => i.ident.name,
            Node::ForeignItem(fi) => fi.ident.name,
            Node::ImplItem(ii) => ii.ident.name,
            Node::TraitItem(ti) => ti.ident.name,
            Node::Variant(v) => v.node.ident.name,
            Node::Field(f) => f.ident.name,
            Node::Lifetime(lt) => lt.name.ident().name,
            Node::GenericParam(param) => param.name.ident().name,
            Node::Binding(&Pat { node: PatKind::Binding(_, _, l, _), .. }) => l.name,
            Node::Ctor(..) => self.name(self.get_parent_item(id)),
            _ => bug!("no name for {}", self.node_to_string(id))
        }
    }

    /// Given a node ID, gets a list of attributes associated with the AST
    /// corresponding to the node-ID.
    pub fn attrs(&self, id: HirId) -> &'hir [ast::Attribute] {
        self.read(id); // reveals attributes on the node
        let attrs = match self.find_entry(id).map(|entry| entry.node) {
            Some(Node::Local(l)) => Some(&l.attrs[..]),
            Some(Node::Item(i)) => Some(&i.attrs[..]),
            Some(Node::ForeignItem(fi)) => Some(&fi.attrs[..]),
            Some(Node::TraitItem(ref ti)) => Some(&ti.attrs[..]),
            Some(Node::ImplItem(ref ii)) => Some(&ii.attrs[..]),
            Some(Node::Variant(ref v)) => Some(&v.node.attrs[..]),
            Some(Node::Field(ref f)) => Some(&f.attrs[..]),
            Some(Node::Expr(ref e)) => Some(&*e.attrs),
            Some(Node::Stmt(ref s)) => Some(s.node.attrs()),
            Some(Node::Arm(ref a)) => Some(&*a.attrs),
            Some(Node::GenericParam(param)) => Some(&param.attrs[..]),
            // Unit/tuple structs/variants take the attributes straight from
            // the struct/variant definition.
            Some(Node::Ctor(..)) => return self.attrs(self.get_parent_item(id)),
            Some(Node::Crate) => Some(&self.forest.krate.attrs[..]),
            _ => None
        };
        attrs.unwrap_or(&[])
    }

    /// Returns an iterator that yields all the hir ids in the map.
    fn all_ids<'a>(&'a self) -> impl Iterator<Item = HirId> + 'a {
        // This code is a bit awkward because the map is implemented as 2 levels of arrays,
        // see the comment on `HirEntryMap`.
        // Iterate over all the indices and return a reference to
        // local maps and their index given that they exist.
        self.map.iter().enumerate().filter_map(|(i, local_map)| {
            local_map.as_ref().map(|m| (i, m))
        }).flat_map(move |(array_index, local_map)| {
            // Iterate over each valid entry in the local map
            local_map.iter_enumerated().filter_map(move |(i, entry)| entry.map(move |_| {
                // Reconstruct the HirId based on the 3 indices we used to find it
                HirId {
                    owner: DefIndex::from(array_index),
                    local_id: i,
                }
            }))
        })
    }

    /// Returns an iterator that yields the node id's with paths that
    /// match `parts`.  (Requires `parts` is non-empty.)
    ///
    /// For example, if given `parts` equal to `["bar", "quux"]`, then
    /// the iterator will produce node id's for items with paths
    /// such as `foo::bar::quux`, `bar::quux`, `other::bar::quux`, and
    /// any other such items it can find in the map.
    pub fn nodes_matching_suffix<'a>(&'a self, parts: &'a [String])
                                 -> impl Iterator<Item = NodeId> + 'a {
        let nodes = NodesMatchingSuffix {
            map: self,
            item_name: parts.last().unwrap(),
            in_which: &parts[..parts.len() - 1],
        };

        self.all_ids().filter(move |hir| nodes.matches_suffix(*hir)).map(move |hir| {
            self.hir_to_node_id(hir)
        })
    }

    pub fn span(&self, hir_id: HirId) -> Span {
        self.read(hir_id); // reveals span from node
        match self.find_entry(hir_id).map(|entry| entry.node) {
            Some(Node::Item(item)) => item.span,
            Some(Node::ForeignItem(foreign_item)) => foreign_item.span,
            Some(Node::TraitItem(trait_method)) => trait_method.span,
            Some(Node::ImplItem(impl_item)) => impl_item.span,
            Some(Node::Variant(variant)) => variant.span,
            Some(Node::Field(field)) => field.span,
            Some(Node::AnonConst(constant)) => self.body(constant.body).value.span,
            Some(Node::Expr(expr)) => expr.span,
            Some(Node::Stmt(stmt)) => stmt.span,
            Some(Node::PathSegment(seg)) => seg.ident.span,
            Some(Node::Ty(ty)) => ty.span,
            Some(Node::TraitRef(tr)) => tr.path.span,
            Some(Node::Binding(pat)) => pat.span,
            Some(Node::Pat(pat)) => pat.span,
            Some(Node::Arm(arm)) => arm.span,
            Some(Node::Block(block)) => block.span,
            Some(Node::Ctor(..)) => match self.find(
                self.get_parent_node(hir_id))
            {
                Some(Node::Item(item)) => item.span,
                Some(Node::Variant(variant)) => variant.span,
                _ => unreachable!(),
            }
            Some(Node::Lifetime(lifetime)) => lifetime.span,
            Some(Node::GenericParam(param)) => param.span,
            Some(Node::Visibility(&Spanned {
                node: VisibilityKind::Restricted { ref path, .. }, ..
            })) => path.span,
            Some(Node::Visibility(v)) => bug!("unexpected Visibility {:?}", v),
            Some(Node::Local(local)) => local.span,
            Some(Node::MacroDef(macro_def)) => macro_def.span,
            Some(Node::Crate) => self.forest.krate.span,
            None => bug!("hir::map::Map::span: id not in map: {:?}", hir_id),
        }
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        self.as_local_hir_id(id).map(|id| self.span(id))
    }

    pub fn node_to_string(&self, id: HirId) -> String {
        hir_id_to_string(self, id, true)
    }

    pub fn hir_to_user_string(&self, id: HirId) -> String {
        hir_id_to_string(self, id, false)
    }

    pub fn hir_to_pretty_string(&self, id: HirId) -> String {
        print::to_string(self, |s| s.print_node(self.get(id)))
    }
}

pub struct NodesMatchingSuffix<'a> {
    map: &'a Map<'a>,
    item_name: &'a String,
    in_which: &'a [String],
}

impl<'a> NodesMatchingSuffix<'a> {
    /// Returns `true` only if some suffix of the module path for parent
    /// matches `self.in_which`.
    ///
    /// In other words: let `[x_0,x_1,...,x_k]` be `self.in_which`;
    /// returns true if parent's path ends with the suffix
    /// `x_0::x_1::...::x_k`.
    fn suffix_matches(&self, parent: HirId) -> bool {
        let mut cursor = parent;
        for part in self.in_which.iter().rev() {
            let (mod_id, mod_name) = match find_first_mod_parent(self.map, cursor) {
                None => return false,
                Some((node_id, name)) => (node_id, name),
            };
            if mod_name.as_str() != *part {
                return false;
            }
            cursor = self.map.get_parent_item(mod_id);
        }
        return true;

        // Finds the first mod in parent chain for `id`, along with
        // that mod's name.
        //
        // If `id` itself is a mod named `m` with parent `p`, then
        // returns `Some(id, m, p)`.  If `id` has no mod in its parent
        // chain, then returns `None`.
        fn find_first_mod_parent(map: &Map<'_>, mut id: HirId) -> Option<(HirId, Name)> {
            loop {
                if let Node::Item(item) = map.find(id)? {
                    if item_is_mod(&item) {
                        return Some((id, item.ident.name))
                    }
                }
                let parent = map.get_parent_item(id);
                if parent == id { return None }
                id = parent;
            }

            fn item_is_mod(item: &Item) -> bool {
                match item.node {
                    ItemKind::Mod(_) => true,
                    _ => false,
                }
            }
        }
    }

    // We are looking at some node `n` with a given name and parent
    // id; do their names match what I am seeking?
    fn matches_names(&self, parent_of_n: HirId, name: Name) -> bool {
        name.as_str() == *self.item_name && self.suffix_matches(parent_of_n)
    }

    fn matches_suffix(&self, hir: HirId) -> bool {
        let name = match self.map.find_entry(hir).map(|entry| entry.node) {
            Some(Node::Item(n)) => n.name(),
            Some(Node::ForeignItem(n)) => n.name(),
            Some(Node::TraitItem(n)) => n.name(),
            Some(Node::ImplItem(n)) => n.name(),
            Some(Node::Variant(n)) => n.name(),
            Some(Node::Field(n)) => n.name(),
            _ => return false,
        };
        self.matches_names(self.map.get_parent_item(hir), name)
    }
}

trait Named {
    fn name(&self) -> Name;
}

impl<T:Named> Named for Spanned<T> { fn name(&self) -> Name { self.node.name() } }

impl Named for Item { fn name(&self) -> Name { self.ident.name } }
impl Named for ForeignItem { fn name(&self) -> Name { self.ident.name } }
impl Named for VariantKind { fn name(&self) -> Name { self.ident.name } }
impl Named for StructField { fn name(&self) -> Name { self.ident.name } }
impl Named for TraitItem { fn name(&self) -> Name { self.ident.name } }
impl Named for ImplItem { fn name(&self) -> Name { self.ident.name } }

pub fn map_crate<'hir>(sess: &crate::session::Session,
                       cstore: &CrateStoreDyn,
                       forest: &'hir Forest,
                       definitions: &'hir Definitions)
                       -> Map<'hir> {
    // Build the reverse mapping of `node_to_hir_id`.
    let hir_to_node_id = definitions.node_to_hir_id.iter_enumerated()
        .map(|(node_id, &hir_id)| (hir_id, node_id)).collect();

    let (map, crate_hash) = {
        let hcx = crate::ich::StableHashingContext::new(sess, &forest.krate, definitions, cstore);

        let mut collector = NodeCollector::root(sess,
                                                &forest.krate,
                                                &forest.dep_graph,
                                                &definitions,
                                                &hir_to_node_id,
                                                hcx);
        intravisit::walk_crate(&mut collector, &forest.krate);

        let crate_disambiguator = sess.local_crate_disambiguator();
        let cmdline_args = sess.opts.dep_tracking_hash();
        collector.finalize_and_compute_crate_hash(
            crate_disambiguator,
            cstore,
            cmdline_args
        )
    };

    let map = Map {
        forest,
        dep_graph: forest.dep_graph.clone(),
        crate_hash,
        map,
        hir_to_node_id,
        definitions,
    };

    time(sess, "validate hir map", || {
        hir_id_validator::check_crate(&map);
    });

    map
}

/// Identical to the `PpAnn` implementation for `hir::Crate`,
/// except it avoids creating a dependency on the whole crate.
impl<'hir> print::PpAnn for Map<'hir> {
    fn nested(&self, state: &mut print::State<'_>, nested: print::Nested) {
        match nested {
            Nested::Item(id) => state.print_item(self.expect_item(id.id)),
            Nested::TraitItem(id) => state.print_trait_item(self.trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.impl_item(id)),
            Nested::Body(id) => state.print_expr(&self.body(id).value),
            Nested::BodyArgPat(id, i) => state.print_pat(&self.body(id).arguments[i].pat)
        }
    }
}

impl<'a> print::State<'a> {
    pub fn print_node(&mut self, node: Node<'_>) {
        match node {
            Node::Item(a)         => self.print_item(&a),
            Node::ForeignItem(a)  => self.print_foreign_item(&a),
            Node::TraitItem(a)    => self.print_trait_item(a),
            Node::ImplItem(a)     => self.print_impl_item(a),
            Node::Variant(a)      => self.print_variant(&a),
            Node::AnonConst(a)    => self.print_anon_const(&a),
            Node::Expr(a)         => self.print_expr(&a),
            Node::Stmt(a)         => self.print_stmt(&a),
            Node::PathSegment(a)  => self.print_path_segment(&a),
            Node::Ty(a)           => self.print_type(&a),
            Node::TraitRef(a)     => self.print_trait_ref(&a),
            Node::Binding(a)      |
            Node::Pat(a)          => self.print_pat(&a),
            Node::Arm(a)          => self.print_arm(&a),
            Node::Block(a)        => {
                use syntax::print::pprust::PrintState;

                // containing cbox, will be closed by print-block at }
                self.cbox(print::indent_unit);
                // head-ibox, will be closed by print-block after {
                self.ibox(0);
                self.print_block(&a)
            }
            Node::Lifetime(a)     => self.print_lifetime(&a),
            Node::Visibility(a)   => self.print_visibility(&a),
            Node::GenericParam(_) => bug!("cannot print Node::GenericParam"),
            Node::Field(_)        => bug!("cannot print StructField"),
            // these cases do not carry enough information in the
            // hir_map to reconstruct their full structure for pretty
            // printing.
            Node::Ctor(..)        => bug!("cannot print isolated Ctor"),
            Node::Local(a)        => self.print_local_decl(&a),
            Node::MacroDef(_)     => bug!("cannot print MacroDef"),
            Node::Crate           => bug!("cannot print Crate"),
        }
    }
}

fn hir_id_to_string(map: &Map<'_>, id: HirId, include_id: bool) -> String {
    let id_str = format!(" (hir_id={})", id);
    let id_str = if include_id { &id_str[..] } else { "" };

    let path_str = || {
        // This functionality is used for debugging, try to use TyCtxt to get
        // the user-friendly path, otherwise fall back to stringifying DefPath.
        crate::ty::tls::with_opt(|tcx| {
            if let Some(tcx) = tcx {
                let def_id = map.local_def_id_from_hir_id(id);
                tcx.def_path_str(def_id)
            } else if let Some(path) = map.def_path_from_hir_id(id) {
                path.data.into_iter().map(|elem| {
                    elem.data.to_string()
                }).collect::<Vec<_>>().join("::")
            } else {
                String::from("<missing path>")
            }
        })
    };

    match map.find(id) {
        Some(Node::Item(item)) => {
            let item_str = match item.node {
                ItemKind::ExternCrate(..) => "extern crate",
                ItemKind::Use(..) => "use",
                ItemKind::Static(..) => "static",
                ItemKind::Const(..) => "const",
                ItemKind::Fn(..) => "fn",
                ItemKind::Mod(..) => "mod",
                ItemKind::ForeignMod(..) => "foreign mod",
                ItemKind::GlobalAsm(..) => "global asm",
                ItemKind::Ty(..) => "ty",
                ItemKind::Existential(..) => "existential type",
                ItemKind::Enum(..) => "enum",
                ItemKind::Struct(..) => "struct",
                ItemKind::Union(..) => "union",
                ItemKind::Trait(..) => "trait",
                ItemKind::TraitAlias(..) => "trait alias",
                ItemKind::Impl(..) => "impl",
            };
            format!("{} {}{}", item_str, path_str(), id_str)
        }
        Some(Node::ForeignItem(_)) => {
            format!("foreign item {}{}", path_str(), id_str)
        }
        Some(Node::ImplItem(ii)) => {
            match ii.node {
                ImplItemKind::Const(..) => {
                    format!("assoc const {} in {}{}", ii.ident, path_str(), id_str)
                }
                ImplItemKind::Method(..) => {
                    format!("method {} in {}{}", ii.ident, path_str(), id_str)
                }
                ImplItemKind::Type(_) => {
                    format!("assoc type {} in {}{}", ii.ident, path_str(), id_str)
                }
                ImplItemKind::Existential(_) => {
                    format!("assoc existential type {} in {}{}", ii.ident, path_str(), id_str)
                }
            }
        }
        Some(Node::TraitItem(ti)) => {
            let kind = match ti.node {
                TraitItemKind::Const(..) => "assoc constant",
                TraitItemKind::Method(..) => "trait method",
                TraitItemKind::Type(..) => "assoc type",
            };

            format!("{} {} in {}{}", kind, ti.ident, path_str(), id_str)
        }
        Some(Node::Variant(ref variant)) => {
            format!("variant {} in {}{}",
                    variant.node.ident,
                    path_str(), id_str)
        }
        Some(Node::Field(ref field)) => {
            format!("field {} in {}{}",
                    field.ident,
                    path_str(), id_str)
        }
        Some(Node::AnonConst(_)) => {
            format!("const {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Expr(_)) => {
            format!("expr {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Stmt(_)) => {
            format!("stmt {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::PathSegment(_)) => {
            format!("path segment {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Ty(_)) => {
            format!("type {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::TraitRef(_)) => {
            format!("trait_ref {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Binding(_)) => {
            format!("local {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Pat(_)) => {
            format!("pat {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Arm(_)) => {
            format!("arm {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Block(_)) => {
            format!("block {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Local(_)) => {
            format!("local {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Ctor(..)) => {
            format!("ctor {}{}", path_str(), id_str)
        }
        Some(Node::Lifetime(_)) => {
            format!("lifetime {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::GenericParam(ref param)) => {
            format!("generic_param {:?}{}", param, id_str)
        }
        Some(Node::Visibility(ref vis)) => {
            format!("visibility {:?}{}", vis, id_str)
        }
        Some(Node::MacroDef(_)) => {
            format!("macro {}{}",  path_str(), id_str)
        }
        Some(Node::Crate) => String::from("root_crate"),
        None => format!("unknown node{}", id_str),
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.def_kind = |tcx, def_id| {
        if let Some(hir_id) = tcx.hir().as_local_hir_id(def_id) {
            tcx.hir().def_kind(hir_id)
        } else {
            bug!("calling local def_kind query provider for upstream DefId: {:?}",
                def_id
            );
        }
    };
}
