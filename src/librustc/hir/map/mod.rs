use self::collector::NodeCollector;
pub use self::definitions::{
    DefKey, DefPath, DefPathData, DefPathHash, Definitions, DisambiguatedDefPathData,
};

use crate::dep_graph::{DepGraph, DepKind, DepNode, DepNodeIndex};
use crate::middle::cstore::CrateStoreDyn;
use crate::ty::query::Providers;
use rustc_ast::ast::{self, Name, NodeId};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::svh::Svh;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::intravisit;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::print::Nested;
use rustc_hir::*;
use rustc_index::vec::IndexVec;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::kw;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

pub mod blocks;
mod collector;
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

    fn fn_decl(&self) -> Option<&'hir FnDecl<'hir>> {
        match self.node {
            Node::Item(ref item) => match item.kind {
                ItemKind::Fn(ref sig, _, _) => Some(&sig.decl),
                _ => None,
            },

            Node::TraitItem(ref item) => match item.kind {
                TraitItemKind::Fn(ref sig, _) => Some(&sig.decl),
                _ => None,
            },

            Node::ImplItem(ref item) => match item.kind {
                ImplItemKind::Method(ref sig, _) => Some(&sig.decl),
                _ => None,
            },

            Node::Expr(ref expr) => match expr.kind {
                ExprKind::Closure(_, ref fn_decl, ..) => Some(fn_decl),
                _ => None,
            },

            _ => None,
        }
    }

    fn fn_sig(&self) -> Option<&'hir FnSig<'hir>> {
        match &self.node {
            Node::Item(item) => match &item.kind {
                ItemKind::Fn(sig, _, _) => Some(sig),
                _ => None,
            },

            Node::TraitItem(item) => match &item.kind {
                TraitItemKind::Fn(sig, _) => Some(sig),
                _ => None,
            },

            Node::ImplItem(item) => match &item.kind {
                ImplItemKind::Method(sig, _) => Some(sig),
                _ => None,
            },

            _ => None,
        }
    }

    fn associated_body(self) -> Option<BodyId> {
        match self.node {
            Node::Item(item) => match item.kind {
                ItemKind::Const(_, body) | ItemKind::Static(.., body) | ItemKind::Fn(.., body) => {
                    Some(body)
                }
                _ => None,
            },

            Node::TraitItem(item) => match item.kind {
                TraitItemKind::Const(_, Some(body))
                | TraitItemKind::Fn(_, TraitMethod::Provided(body)) => Some(body),
                _ => None,
            },

            Node::ImplItem(item) => match item.kind {
                ImplItemKind::Const(_, body) | ImplItemKind::Method(_, body) => Some(body),
                _ => None,
            },

            Node::AnonConst(constant) => Some(constant.body),

            Node::Expr(expr) => match expr.kind {
                ExprKind::Closure(.., body, _, _) => Some(body),
                _ => None,
            },

            _ => None,
        }
    }

    fn is_body_owner(self, hir_id: HirId) -> bool {
        match self.associated_body() {
            Some(b) => b.hir_id == hir_id,
            None => false,
        }
    }
}

/// This type is effectively a `HashMap<HirId, Entry<'hir>>`,
/// but it is implemented as 2 layers of arrays.
/// - first we have `A = IndexVec<DefIndex, B>` mapping `DefIndex`s to an inner value
/// - which is `B = IndexVec<ItemLocalId, Option<Entry<'hir>>` which gives you the `Entry`.
pub(super) type HirEntryMap<'hir> = IndexVec<DefIndex, IndexVec<ItemLocalId, Option<Entry<'hir>>>>;

/// Represents a mapping from `NodeId`s to AST elements and their parent `NodeId`s.
#[derive(Clone)]
pub struct Map<'hir> {
    krate: &'hir Crate<'hir>,

    pub dep_graph: DepGraph,

    /// The SVH of the local crate.
    pub crate_hash: Svh,

    map: HirEntryMap<'hir>,

    definitions: Definitions,

    /// The reverse mapping of `node_to_hir_id`.
    hir_to_node_id: FxHashMap<HirId, NodeId>,
}

/// An iterator that walks up the ancestor tree of a given `HirId`.
/// Constructed using `tcx.hir().parent_iter(hir_id)`.
pub struct ParentHirIterator<'map, 'hir> {
    current_id: HirId,
    map: &'map Map<'hir>,
}

impl<'hir> Iterator for ParentHirIterator<'_, 'hir> {
    type Item = (HirId, Node<'hir>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id == CRATE_HIR_ID {
            return None;
        }
        loop {
            // There are nodes that do not have entries, so we need to skip them.
            let parent_id = self.map.get_parent_node(self.current_id);

            if parent_id == self.current_id {
                self.current_id = CRATE_HIR_ID;
                return None;
            }

            self.current_id = parent_id;
            if let Some(entry) = self.map.find_entry(parent_id) {
                return Some((parent_id, entry.node));
            }
            // If this `HirId` doesn't have an `Entry`, skip it and look for its `parent_id`.
        }
    }
}

impl<'hir> Map<'hir> {
    /// This is used internally in the dependency tracking system.
    /// Use the `krate` method to ensure your dependency on the
    /// crate is tracked.
    pub fn untracked_krate(&self) -> &Crate<'hir> {
        &self.krate
    }

    #[inline]
    fn lookup(&self, id: HirId) -> Option<&Entry<'hir>> {
        let local_map = self.map.get(id.owner)?;
        local_map.get(id.local_id)?.as_ref()
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
    pub fn definitions(&self) -> &Definitions {
        &self.definitions
    }

    pub fn def_key(&self, def_id: DefId) -> DefKey {
        assert!(def_id.is_local());
        self.definitions.def_key(def_id.index)
    }

    pub fn def_path_from_hir_id(&self, id: HirId) -> Option<DefPath> {
        self.opt_local_def_id(id).map(|def_id| self.def_path(def_id))
    }

    pub fn def_path(&self, def_id: DefId) -> DefPath {
        assert!(def_id.is_local());
        self.definitions.def_path(def_id.index)
    }

    #[inline]
    pub fn local_def_id_from_node_id(&self, node: NodeId) -> DefId {
        self.opt_local_def_id_from_node_id(node).unwrap_or_else(|| {
            let hir_id = self.node_to_hir_id(node);
            bug!(
                "local_def_id_from_node_id: no entry for `{}`, which has a map of `{:?}`",
                node,
                self.find_entry(hir_id)
            )
        })
    }

    #[inline]
    pub fn local_def_id(&self, hir_id: HirId) -> DefId {
        self.opt_local_def_id(hir_id).unwrap_or_else(|| {
            bug!(
                "local_def_id: no entry for `{:?}`, which has a map of `{:?}`",
                hir_id,
                self.find_entry(hir_id)
            )
        })
    }

    #[inline]
    pub fn opt_local_def_id(&self, hir_id: HirId) -> Option<DefId> {
        let node_id = self.hir_to_node_id(hir_id);
        self.definitions.opt_local_def_id(node_id)
    }

    #[inline]
    pub fn opt_local_def_id_from_node_id(&self, node: NodeId) -> Option<DefId> {
        self.definitions.opt_local_def_id(node)
    }

    #[inline]
    pub fn as_local_node_id(&self, def_id: DefId) -> Option<NodeId> {
        self.definitions.as_local_node_id(def_id)
    }

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

    pub fn def_kind(&self, hir_id: HirId) -> Option<DefKind> {
        let node = self.find(hir_id)?;

        Some(match node {
            Node::Item(item) => match item.kind {
                ItemKind::Static(..) => DefKind::Static,
                ItemKind::Const(..) => DefKind::Const,
                ItemKind::Fn(..) => DefKind::Fn,
                ItemKind::Mod(..) => DefKind::Mod,
                ItemKind::OpaqueTy(..) => DefKind::OpaqueTy,
                ItemKind::TyAlias(..) => DefKind::TyAlias,
                ItemKind::Enum(..) => DefKind::Enum,
                ItemKind::Struct(..) => DefKind::Struct,
                ItemKind::Union(..) => DefKind::Union,
                ItemKind::Trait(..) => DefKind::Trait,
                ItemKind::TraitAlias(..) => DefKind::TraitAlias,
                ItemKind::ExternCrate(_)
                | ItemKind::Use(..)
                | ItemKind::ForeignMod(..)
                | ItemKind::GlobalAsm(..)
                | ItemKind::Impl { .. } => return None,
            },
            Node::ForeignItem(item) => match item.kind {
                ForeignItemKind::Fn(..) => DefKind::Fn,
                ForeignItemKind::Static(..) => DefKind::Static,
                ForeignItemKind::Type => DefKind::ForeignTy,
            },
            Node::TraitItem(item) => match item.kind {
                TraitItemKind::Const(..) => DefKind::AssocConst,
                TraitItemKind::Fn(..) => DefKind::AssocFn,
                TraitItemKind::Type(..) => DefKind::AssocTy,
            },
            Node::ImplItem(item) => match item.kind {
                ImplItemKind::Const(..) => DefKind::AssocConst,
                ImplItemKind::Method(..) => DefKind::AssocFn,
                ImplItemKind::TyAlias(..) => DefKind::AssocTy,
                ImplItemKind::OpaqueTy(..) => DefKind::AssocOpaqueTy,
            },
            Node::Variant(_) => DefKind::Variant,
            Node::Ctor(variant_data) => {
                // FIXME(eddyb) is this even possible, if we have a `Node::Ctor`?
                variant_data.ctor_hir_id()?;

                let ctor_of = match self.find(self.get_parent_node(hir_id)) {
                    Some(Node::Item(..)) => def::CtorOf::Struct,
                    Some(Node::Variant(..)) => def::CtorOf::Variant,
                    _ => unreachable!(),
                };
                DefKind::Ctor(ctor_of, def::CtorKind::from_hir(variant_data))
            }
            Node::AnonConst(_)
            | Node::Field(_)
            | Node::Expr(_)
            | Node::Stmt(_)
            | Node::PathSegment(_)
            | Node::Ty(_)
            | Node::TraitRef(_)
            | Node::Pat(_)
            | Node::Binding(_)
            | Node::Local(_)
            | Node::Param(_)
            | Node::Arm(_)
            | Node::Lifetime(_)
            | Node::Visibility(_)
            | Node::Block(_)
            | Node::Crate => return None,
            Node::MacroDef(_) => DefKind::Macro(MacroKind::Bang),
            Node::GenericParam(param) => match param.kind {
                GenericParamKind::Lifetime { .. } => return None,
                GenericParamKind::Type { .. } => DefKind::TyParam,
                GenericParamKind::Const { .. } => DefKind::ConstParam,
            },
        })
    }

    fn find_entry(&self, id: HirId) -> Option<Entry<'hir>> {
        self.lookup(id).cloned()
    }

    pub fn item(&self, id: HirId) -> &'hir Item<'hir> {
        self.read(id);

        // N.B., intentionally bypass `self.krate()` so that we
        // do not trigger a read of the whole krate here
        self.krate.item(id)
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.krate()` so that we
        // do not trigger a read of the whole krate here
        self.krate.trait_item(id)
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.krate()` so that we
        // do not trigger a read of the whole krate here
        self.krate.impl_item(id)
    }

    pub fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        self.read(id.hir_id);

        // N.B., intentionally bypass `self.krate()` so that we
        // do not trigger a read of the whole krate here
        self.krate.body(id)
    }

    pub fn fn_decl_by_hir_id(&self, hir_id: HirId) -> Option<&'hir FnDecl<'hir>> {
        if let Some(entry) = self.find_entry(hir_id) {
            entry.fn_decl()
        } else {
            bug!("no entry for hir_id `{}`", hir_id)
        }
    }

    pub fn fn_sig_by_hir_id(&self, hir_id: HirId) -> Option<&'hir FnSig<'hir>> {
        if let Some(entry) = self.find_entry(hir_id) {
            entry.fn_sig()
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
        self.local_def_id(self.body_owner(id))
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
            span_bug!(
                self.span(id),
                "body_owned_by: {} has no associated body",
                self.node_to_string(id)
            );
        })
    }

    pub fn body_owner_kind(&self, id: HirId) -> BodyOwnerKind {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Const(..), .. })
            | Node::TraitItem(&TraitItem { kind: TraitItemKind::Const(..), .. })
            | Node::ImplItem(&ImplItem { kind: ImplItemKind::Const(..), .. })
            | Node::AnonConst(_) => BodyOwnerKind::Const,
            Node::Ctor(..)
            | Node::Item(&Item { kind: ItemKind::Fn(..), .. })
            | Node::TraitItem(&TraitItem { kind: TraitItemKind::Fn(..), .. })
            | Node::ImplItem(&ImplItem { kind: ImplItemKind::Method(..), .. }) => BodyOwnerKind::Fn,
            Node::Item(&Item { kind: ItemKind::Static(_, m, _), .. }) => BodyOwnerKind::Static(m),
            Node::Expr(&Expr { kind: ExprKind::Closure(..), .. }) => BodyOwnerKind::Closure,
            node => bug!("{:#?} is not a body node", node),
        }
    }

    pub fn ty_param_owner(&self, id: HirId) -> HirId {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Trait(..), .. })
            | Node::Item(&Item { kind: ItemKind::TraitAlias(..), .. }) => id,
            Node::GenericParam(_) => self.get_parent_node(id),
            _ => bug!("ty_param_owner: {} not a type parameter", self.node_to_string(id)),
        }
    }

    pub fn ty_param_name(&self, id: HirId) -> Name {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Trait(..), .. })
            | Node::Item(&Item { kind: ItemKind::TraitAlias(..), .. }) => kw::SelfUpper,
            Node::GenericParam(param) => param.name.ident().name,
            _ => bug!("ty_param_name: {} not a type parameter", self.node_to_string(id)),
        }
    }

    pub fn trait_impls(&self, trait_did: DefId) -> &'hir [HirId] {
        self.dep_graph.read(DepNode::new_no_params(DepKind::AllLocalTraitImpls));

        // N.B., intentionally bypass `self.krate()` so that we
        // do not trigger a read of the whole krate here
        self.krate.trait_impls.get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    /// Gets the attributes on the crate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'hir [ast::Attribute] {
        let def_path_hash = self.definitions.def_path_hash(CRATE_DEF_INDEX);

        self.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
        &self.krate.attrs
    }

    pub fn get_module(&self, module: DefId) -> (&'hir Mod<'hir>, Span, HirId) {
        let hir_id = self.as_local_hir_id(module).unwrap();
        self.read(hir_id);
        match self.find_entry(hir_id).unwrap().node {
            Node::Item(&Item { span, kind: ItemKind::Mod(ref m), .. }) => (m, span, hir_id),
            Node::Crate => (&self.krate.module, self.krate.span, hir_id),
            node => panic!("not a module: {:?}", node),
        }
    }

    pub fn visit_item_likes_in_module<V>(&self, module: DefId, visitor: &mut V)
    where
        V: ItemLikeVisitor<'hir>,
    {
        let hir_id = self.as_local_hir_id(module).unwrap();

        // Read the module so we'll be re-executed if new items
        // appear immediately under in the module. If some new item appears
        // in some nested item in the module, we'll be re-executed due to reads
        // in the expect_* calls the loops below
        self.read(hir_id);

        let module = &self.krate.modules[&hir_id];

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
        self.find(id).unwrap_or_else(|| bug!("couldn't find hir id {} in the HIR map", id))
    }

    pub fn get_if_local(&self, id: DefId) -> Option<Node<'hir>> {
        self.as_local_hir_id(id).map(|id| self.get(id)) // read recorded by `get`
    }

    pub fn get_generics(&self, id: DefId) -> Option<&'hir Generics<'hir>> {
        self.get_if_local(id).and_then(|node| match node {
            Node::ImplItem(ref impl_item) => Some(&impl_item.generics),
            Node::TraitItem(ref trait_item) => Some(&trait_item.generics),
            Node::Item(ref item) => match item.kind {
                ItemKind::Fn(_, ref generics, _)
                | ItemKind::TyAlias(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics)
                | ItemKind::Trait(_, _, ref generics, ..)
                | ItemKind::TraitAlias(ref generics, _)
                | ItemKind::Impl { ref generics, .. } => Some(generics),
                _ => None,
            },
            _ => None,
        })
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    pub fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        let result = self
            .find_entry(hir_id)
            .and_then(|entry| if let Node::Crate = entry.node { None } else { Some(entry.node) });
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

        self.find_entry(hir_id).and_then(|x| x.parent_node()).unwrap_or(hir_id)
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `get_parent_node`.
    pub fn parent_iter(&self, current_id: HirId) -> ParentHirIterator<'_, 'hir> {
        ParentHirIterator { current_id, map: self }
    }

    /// Checks if the node is an argument. An argument is a local variable whose
    /// immediate parent is an item or a closure.
    pub fn is_argument(&self, id: HirId) -> bool {
        match self.find(id) {
            Some(Node::Binding(_)) => (),
            _ => return false,
        }
        match self.find(self.get_parent_node(id)) {
            Some(Node::Item(_)) | Some(Node::TraitItem(_)) | Some(Node::ImplItem(_)) => true,
            Some(Node::Expr(e)) => match e.kind {
                ExprKind::Closure(..) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether the expression pointed at by `hir_id` belongs to a `const` evaluation context.
    /// Used exclusively for diagnostics, to avoid suggestion function calls.
    pub fn is_const_context(&self, hir_id: HirId) -> bool {
        let parent_id = self.get_parent_item(hir_id);
        match self.get(parent_id) {
            Node::Item(&Item { kind: ItemKind::Const(..), .. })
            | Node::TraitItem(&TraitItem { kind: TraitItemKind::Const(..), .. })
            | Node::ImplItem(&ImplItem { kind: ImplItemKind::Const(..), .. })
            | Node::AnonConst(_)
            | Node::Item(&Item { kind: ItemKind::Static(..), .. }) => true,
            Node::Item(&Item { kind: ItemKind::Fn(ref sig, ..), .. }) => {
                sig.header.constness == Constness::Const
            }
            _ => false,
        }
    }

    /// Whether `hir_id` corresponds to a `mod` or a crate.
    pub fn is_hir_id_module(&self, hir_id: HirId) -> bool {
        match self.lookup(hir_id) {
            Some(Entry { node: Node::Item(Item { kind: ItemKind::Mod(_), .. }), .. })
            | Some(Entry { node: Node::Crate, .. }) => true,
            _ => false,
        }
    }

    /// Retrieves the `HirId` for `id`'s enclosing method, unless there's a
    /// `while` or `loop` before reaching it, as block tail returns are not
    /// available in them.
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     if x == 1 {
    ///         true  // If `get_return_block` gets passed the `id` corresponding
    ///     } else {  // to this, it will return `foo`'s `HirId`.
    ///         false
    ///     }
    /// }
    /// ```
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     loop {
    ///         true  // If `get_return_block` gets passed the `id` corresponding
    ///     }         // to this, it will return `None`.
    ///     false
    /// }
    /// ```
    pub fn get_return_block(&self, id: HirId) -> Option<HirId> {
        let mut iter = self.parent_iter(id).peekable();
        let mut ignore_tail = false;
        if let Some(entry) = self.find_entry(id) {
            if let Node::Expr(Expr { kind: ExprKind::Ret(_), .. }) = entry.node {
                // When dealing with `return` statements, we don't care about climbing only tail
                // expressions.
                ignore_tail = true;
            }
        }
        while let Some((hir_id, node)) = iter.next() {
            if let (Some((_, next_node)), false) = (iter.peek(), ignore_tail) {
                match next_node {
                    Node::Block(Block { expr: None, .. }) => return None,
                    Node::Block(Block { expr: Some(expr), .. }) => {
                        if hir_id != expr.hir_id {
                            // The current node is not the tail expression of its parent.
                            return None;
                        }
                    }
                    _ => {}
                }
            }
            match node {
                Node::Item(_)
                | Node::ForeignItem(_)
                | Node::TraitItem(_)
                | Node::Expr(Expr { kind: ExprKind::Closure(..), .. })
                | Node::ImplItem(_) => return Some(hir_id),
                Node::Expr(ref expr) => {
                    match expr.kind {
                        // Ignore `return`s on the first iteration
                        ExprKind::Loop(..) | ExprKind::Ret(..) => return None,
                        _ => {}
                    }
                }
                Node::Local(_) => return None,
                _ => {}
            }
        }
        None
    }

    /// Retrieves the `HirId` for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the HIR which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent_item(&self, hir_id: HirId) -> HirId {
        for (hir_id, node) in self.parent_iter(hir_id) {
            match node {
                Node::Crate
                | Node::Item(_)
                | Node::ForeignItem(_)
                | Node::TraitItem(_)
                | Node::ImplItem(_) => return hir_id,
                _ => {}
            }
        }
        hir_id
    }

    /// Returns the `HirId` of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub(super) fn get_module_parent_node(&self, hir_id: HirId) -> HirId {
        for (hir_id, node) in self.parent_iter(hir_id) {
            if let Node::Item(&Item { kind: ItemKind::Mod(_), .. }) = node {
                return hir_id;
            }
        }
        CRATE_HIR_ID
    }

    /// When on a match arm tail expression or on a match arm, give back the enclosing `match`
    /// expression.
    ///
    /// Used by error reporting when there's a type error in a match arm caused by the `match`
    /// expression needing to be unit.
    pub fn get_match_if_cause(&self, hir_id: HirId) -> Option<&'hir Expr<'hir>> {
        for (_, node) in self.parent_iter(hir_id) {
            match node {
                Node::Item(_) | Node::ForeignItem(_) | Node::TraitItem(_) | Node::ImplItem(_) => {
                    break;
                }
                Node::Expr(expr) => match expr.kind {
                    ExprKind::Match(_, _, _) => return Some(expr),
                    _ => {}
                },
                Node::Stmt(stmt) => match stmt.kind {
                    StmtKind::Local(_) => break,
                    _ => {}
                },
                _ => {}
            }
        }
        None
    }

    /// Returns the nearest enclosing scope. A scope is roughly an item or block.
    pub fn get_enclosing_scope(&self, hir_id: HirId) -> Option<HirId> {
        for (hir_id, node) in self.parent_iter(hir_id) {
            if match node {
                Node::Item(i) => match i.kind {
                    ItemKind::Fn(..)
                    | ItemKind::Mod(..)
                    | ItemKind::Enum(..)
                    | ItemKind::Struct(..)
                    | ItemKind::Union(..)
                    | ItemKind::Trait(..)
                    | ItemKind::Impl { .. } => true,
                    _ => false,
                },
                Node::ForeignItem(fi) => match fi.kind {
                    ForeignItemKind::Fn(..) => true,
                    _ => false,
                },
                Node::TraitItem(ti) => match ti.kind {
                    TraitItemKind::Fn(..) => true,
                    _ => false,
                },
                Node::ImplItem(ii) => match ii.kind {
                    ImplItemKind::Method(..) => true,
                    _ => false,
                },
                Node::Block(_) => true,
                _ => false,
            } {
                return Some(hir_id);
            }
        }
        None
    }

    /// Returns the defining scope for an opaque type definition.
    pub fn get_defining_scope(&self, id: HirId) -> HirId {
        let mut scope = id;
        loop {
            scope = self.get_enclosing_scope(scope).unwrap_or(CRATE_HIR_ID);
            if scope == CRATE_HIR_ID {
                return CRATE_HIR_ID;
            }
            match self.get(scope) {
                Node::Item(i) => match i.kind {
                    ItemKind::OpaqueTy(OpaqueTy { impl_trait_fn: None, .. }) => {}
                    _ => break,
                },
                Node::Block(_) => {}
                _ => break,
            }
        }
        scope
    }

    pub fn get_parent_did(&self, id: HirId) -> DefId {
        self.local_def_id(self.get_parent_item(id))
    }

    pub fn get_foreign_abi(&self, hir_id: HirId) -> Abi {
        let parent = self.get_parent_item(hir_id);
        if let Some(entry) = self.find_entry(parent) {
            if let Entry {
                node: Node::Item(Item { kind: ItemKind::ForeignMod(ref nm), .. }), ..
            } = entry
            {
                self.read(hir_id); // reveals some of the content of a node
                return nm.abi;
            }
        }
        bug!("expected foreign mod or inlined parent, found {}", self.node_to_string(parent))
    }

    pub fn expect_item(&self, id: HirId) -> &'hir Item<'hir> {
        match self.find(id) {
            // read recorded by `find`
            Some(Node::Item(item)) => item,
            _ => bug!("expected item, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_impl_item(&self, id: HirId) -> &'hir ImplItem<'hir> {
        match self.find(id) {
            Some(Node::ImplItem(item)) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_trait_item(&self, id: HirId) -> &'hir TraitItem<'hir> {
        match self.find(id) {
            Some(Node::TraitItem(item)) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_variant_data(&self, id: HirId) -> &'hir VariantData<'hir> {
        match self.find(id) {
            Some(Node::Item(i)) => match i.kind {
                ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                    struct_def
                }
                _ => bug!("struct ID bound to non-struct {}", self.node_to_string(id)),
            },
            Some(Node::Variant(variant)) => &variant.data,
            Some(Node::Ctor(data)) => data,
            _ => bug!("expected struct or variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_variant(&self, id: HirId) -> &'hir Variant<'hir> {
        match self.find(id) {
            Some(Node::Variant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(&self, id: HirId) -> &'hir ForeignItem<'hir> {
        match self.find(id) {
            Some(Node::ForeignItem(item)) => item,
            _ => bug!("expected foreign item, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_expr(&self, id: HirId) -> &'hir Expr<'hir> {
        match self.find(id) {
            // read recorded by find
            Some(Node::Expr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id)),
        }
    }

    pub fn opt_name(&self, id: HirId) -> Option<Name> {
        Some(match self.get(id) {
            Node::Item(i) => i.ident.name,
            Node::ForeignItem(fi) => fi.ident.name,
            Node::ImplItem(ii) => ii.ident.name,
            Node::TraitItem(ti) => ti.ident.name,
            Node::Variant(v) => v.ident.name,
            Node::Field(f) => f.ident.name,
            Node::Lifetime(lt) => lt.name.ident().name,
            Node::GenericParam(param) => param.name.ident().name,
            Node::Binding(&Pat { kind: PatKind::Binding(_, _, l, _), .. }) => l.name,
            Node::Ctor(..) => self.name(self.get_parent_item(id)),
            _ => return None,
        })
    }

    pub fn name(&self, id: HirId) -> Name {
        match self.opt_name(id) {
            Some(name) => name,
            None => bug!("no name for {}", self.node_to_string(id)),
        }
    }

    /// Given a node ID, gets a list of attributes associated with the AST
    /// corresponding to the node-ID.
    pub fn attrs(&self, id: HirId) -> &'hir [ast::Attribute] {
        self.read(id); // reveals attributes on the node
        let attrs = match self.find_entry(id).map(|entry| entry.node) {
            Some(Node::Param(a)) => Some(&a.attrs[..]),
            Some(Node::Local(l)) => Some(&l.attrs[..]),
            Some(Node::Item(i)) => Some(&i.attrs[..]),
            Some(Node::ForeignItem(fi)) => Some(&fi.attrs[..]),
            Some(Node::TraitItem(ref ti)) => Some(&ti.attrs[..]),
            Some(Node::ImplItem(ref ii)) => Some(&ii.attrs[..]),
            Some(Node::Variant(ref v)) => Some(&v.attrs[..]),
            Some(Node::Field(ref f)) => Some(&f.attrs[..]),
            Some(Node::Expr(ref e)) => Some(&*e.attrs),
            Some(Node::Stmt(ref s)) => Some(s.kind.attrs()),
            Some(Node::Arm(ref a)) => Some(&*a.attrs),
            Some(Node::GenericParam(param)) => Some(&param.attrs[..]),
            // Unit/tuple structs/variants take the attributes straight from
            // the struct/variant definition.
            Some(Node::Ctor(..)) => return self.attrs(self.get_parent_item(id)),
            Some(Node::Crate) => Some(&self.krate.attrs[..]),
            _ => None,
        };
        attrs.unwrap_or(&[])
    }

    /// Returns an iterator that yields all the hir ids in the map.
    fn all_ids<'a>(&'a self) -> impl Iterator<Item = HirId> + 'a {
        // This code is a bit awkward because the map is implemented as 2 levels of arrays,
        // see the comment on `HirEntryMap`.
        // Iterate over all the indices and return a reference to
        // local maps and their index given that they exist.
        self.map.iter_enumerated().flat_map(move |(owner, local_map)| {
            // Iterate over each valid entry in the local map.
            local_map.iter_enumerated().filter_map(move |(i, entry)| {
                entry.map(move |_| {
                    // Reconstruct the `HirId` based on the 3 indices we used to find it.
                    HirId { owner, local_id: i }
                })
            })
        })
    }

    /// Returns an iterator that yields the node id's with paths that
    /// match `parts`.  (Requires `parts` is non-empty.)
    ///
    /// For example, if given `parts` equal to `["bar", "quux"]`, then
    /// the iterator will produce node id's for items with paths
    /// such as `foo::bar::quux`, `bar::quux`, `other::bar::quux`, and
    /// any other such items it can find in the map.
    pub fn nodes_matching_suffix<'a>(
        &'a self,
        parts: &'a [String],
    ) -> impl Iterator<Item = NodeId> + 'a {
        let nodes = NodesMatchingSuffix {
            map: self,
            item_name: parts.last().unwrap(),
            in_which: &parts[..parts.len() - 1],
        };

        self.all_ids()
            .filter(move |hir| nodes.matches_suffix(*hir))
            .map(move |hir| self.hir_to_node_id(hir))
    }

    pub fn span(&self, hir_id: HirId) -> Span {
        self.read(hir_id); // reveals span from node
        match self.find_entry(hir_id).map(|entry| entry.node) {
            Some(Node::Param(param)) => param.span,
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
            Some(Node::Ctor(..)) => match self.find(self.get_parent_node(hir_id)) {
                Some(Node::Item(item)) => item.span,
                Some(Node::Variant(variant)) => variant.span,
                _ => unreachable!(),
            },
            Some(Node::Lifetime(lifetime)) => lifetime.span,
            Some(Node::GenericParam(param)) => param.span,
            Some(Node::Visibility(&Spanned {
                node: VisibilityKind::Restricted { ref path, .. },
                ..
            })) => path.span,
            Some(Node::Visibility(v)) => bug!("unexpected Visibility {:?}", v),
            Some(Node::Local(local)) => local.span,
            Some(Node::MacroDef(macro_def)) => macro_def.span,
            Some(Node::Crate) => self.krate.span,
            None => bug!("hir::map::Map::span: id not in map: {:?}", hir_id),
        }
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        self.as_local_hir_id(id).map(|id| self.span(id))
    }

    pub fn res_span(&self, res: Res) -> Option<Span> {
        match res {
            Res::Err => None,
            Res::Local(id) => Some(self.span(id)),
            res => self.span_if_local(res.opt_def_id()?),
        }
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

impl<'hir> intravisit::Map<'hir> for Map<'hir> {
    fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        self.body(id)
    }

    fn item(&self, id: HirId) -> &'hir Item<'hir> {
        self.item(id)
    }

    fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.trait_item(id)
    }

    fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.impl_item(id)
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
                        return Some((id, item.ident.name));
                    }
                }
                let parent = map.get_parent_item(id);
                if parent == id {
                    return None;
                }
                id = parent;
            }

            fn item_is_mod(item: &Item<'_>) -> bool {
                match item.kind {
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

impl<T: Named> Named for Spanned<T> {
    fn name(&self) -> Name {
        self.node.name()
    }
}

impl Named for Item<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}
impl Named for ForeignItem<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}
impl Named for Variant<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}
impl Named for StructField<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}
impl Named for TraitItem<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}
impl Named for ImplItem<'_> {
    fn name(&self) -> Name {
        self.ident.name
    }
}

pub fn map_crate<'hir>(
    sess: &rustc_session::Session,
    cstore: &CrateStoreDyn,
    krate: &'hir Crate<'hir>,
    dep_graph: DepGraph,
    definitions: Definitions,
) -> Map<'hir> {
    let _prof_timer = sess.prof.generic_activity("build_hir_map");

    // Build the reverse mapping of `node_to_hir_id`.
    let hir_to_node_id = definitions
        .node_to_hir_id
        .iter_enumerated()
        .map(|(node_id, &hir_id)| (hir_id, node_id))
        .collect();

    let (map, crate_hash) = {
        let hcx = crate::ich::StableHashingContext::new(sess, krate, &definitions, cstore);

        let mut collector =
            NodeCollector::root(sess, krate, &dep_graph, &definitions, &hir_to_node_id, hcx);
        intravisit::walk_crate(&mut collector, krate);

        let crate_disambiguator = sess.local_crate_disambiguator();
        let cmdline_args = sess.opts.dep_tracking_hash();
        collector.finalize_and_compute_crate_hash(crate_disambiguator, cstore, cmdline_args)
    };

    let map = Map { krate, dep_graph, crate_hash, map, hir_to_node_id, definitions };

    sess.time("validate_HIR_map", || {
        hir_id_validator::check_crate(&map, sess);
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
            Nested::BodyParamPat(id, i) => state.print_pat(&self.body(id).params[i].pat),
        }
    }
}

fn hir_id_to_string(map: &Map<'_>, id: HirId, include_id: bool) -> String {
    let id_str = format!(" (hir_id={})", id);
    let id_str = if include_id { &id_str[..] } else { "" };

    let path_str = || {
        // This functionality is used for debugging, try to use `TyCtxt` to get
        // the user-friendly path, otherwise fall back to stringifying `DefPath`.
        crate::ty::tls::with_opt(|tcx| {
            if let Some(tcx) = tcx {
                let def_id = map.local_def_id(id);
                tcx.def_path_str(def_id)
            } else if let Some(path) = map.def_path_from_hir_id(id) {
                path.data
                    .into_iter()
                    .map(|elem| elem.data.to_string())
                    .collect::<Vec<_>>()
                    .join("::")
            } else {
                String::from("<missing path>")
            }
        })
    };

    match map.find(id) {
        Some(Node::Item(item)) => {
            let item_str = match item.kind {
                ItemKind::ExternCrate(..) => "extern crate",
                ItemKind::Use(..) => "use",
                ItemKind::Static(..) => "static",
                ItemKind::Const(..) => "const",
                ItemKind::Fn(..) => "fn",
                ItemKind::Mod(..) => "mod",
                ItemKind::ForeignMod(..) => "foreign mod",
                ItemKind::GlobalAsm(..) => "global asm",
                ItemKind::TyAlias(..) => "ty",
                ItemKind::OpaqueTy(..) => "opaque type",
                ItemKind::Enum(..) => "enum",
                ItemKind::Struct(..) => "struct",
                ItemKind::Union(..) => "union",
                ItemKind::Trait(..) => "trait",
                ItemKind::TraitAlias(..) => "trait alias",
                ItemKind::Impl { .. } => "impl",
            };
            format!("{} {}{}", item_str, path_str(), id_str)
        }
        Some(Node::ForeignItem(_)) => format!("foreign item {}{}", path_str(), id_str),
        Some(Node::ImplItem(ii)) => match ii.kind {
            ImplItemKind::Const(..) => {
                format!("assoc const {} in {}{}", ii.ident, path_str(), id_str)
            }
            ImplItemKind::Method(..) => format!("method {} in {}{}", ii.ident, path_str(), id_str),
            ImplItemKind::TyAlias(_) => {
                format!("assoc type {} in {}{}", ii.ident, path_str(), id_str)
            }
            ImplItemKind::OpaqueTy(_) => {
                format!("assoc opaque type {} in {}{}", ii.ident, path_str(), id_str)
            }
        },
        Some(Node::TraitItem(ti)) => {
            let kind = match ti.kind {
                TraitItemKind::Const(..) => "assoc constant",
                TraitItemKind::Fn(..) => "trait method",
                TraitItemKind::Type(..) => "assoc type",
            };

            format!("{} {} in {}{}", kind, ti.ident, path_str(), id_str)
        }
        Some(Node::Variant(ref variant)) => {
            format!("variant {} in {}{}", variant.ident, path_str(), id_str)
        }
        Some(Node::Field(ref field)) => {
            format!("field {} in {}{}", field.ident, path_str(), id_str)
        }
        Some(Node::AnonConst(_)) => format!("const {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Expr(_)) => format!("expr {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Stmt(_)) => format!("stmt {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::PathSegment(_)) => {
            format!("path segment {}{}", map.hir_to_pretty_string(id), id_str)
        }
        Some(Node::Ty(_)) => format!("type {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::TraitRef(_)) => format!("trait_ref {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Binding(_)) => format!("local {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Pat(_)) => format!("pat {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Param(_)) => format!("param {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Arm(_)) => format!("arm {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Block(_)) => format!("block {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Local(_)) => format!("local {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::Ctor(..)) => format!("ctor {}{}", path_str(), id_str),
        Some(Node::Lifetime(_)) => format!("lifetime {}{}", map.hir_to_pretty_string(id), id_str),
        Some(Node::GenericParam(ref param)) => format!("generic_param {:?}{}", param, id_str),
        Some(Node::Visibility(ref vis)) => format!("visibility {:?}{}", vis, id_str),
        Some(Node::MacroDef(_)) => format!("macro {}{}", path_str(), id_str),
        Some(Node::Crate) => String::from("root_crate"),
        None => format!("unknown node{}", id_str),
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.def_kind = |tcx, def_id| {
        if let Some(hir_id) = tcx.hir().as_local_hir_id(def_id) {
            tcx.hir().def_kind(hir_id)
        } else {
            bug!("calling local def_kind query provider for upstream DefId: {:?}", def_id);
        }
    };
}
