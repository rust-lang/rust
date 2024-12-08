use rustc_abi::ExternAbi;
use rustc_ast::visit::{VisitorResult, walk_list};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{DynSend, DynSync, par_for_each_in, try_par_for_each_in};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId, LocalModDefId};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_hir::intravisit::Visitor;
use rustc_hir::*;
use rustc_middle::hir::nested_filter;
use rustc_span::def_id::StableCrateId;
use rustc_span::symbol::{Ident, Symbol, kw, sym};
use rustc_span::{ErrorGuaranteed, Span};
use {rustc_ast as ast, rustc_hir_pretty as pprust_hir};

use crate::hir::ModuleItems;
use crate::middle::debugger_visualizer::DebuggerVisualizerFile;
use crate::query::LocalCrate;
use crate::ty::TyCtxt;

// FIXME: the structure was necessary in the past but now it
// only serves as "namespace" for HIR-related methods, and can be
// removed if all the methods are reasonably renamed and moved to tcx
// (https://github.com/rust-lang/rust/pull/118256#issuecomment-1826442834).
#[derive(Copy, Clone)]
pub struct Map<'hir> {
    pub(super) tcx: TyCtxt<'hir>,
}

/// An iterator that walks up the ancestor tree of a given `HirId`.
/// Constructed using `tcx.hir().parent_iter(hir_id)`.
struct ParentHirIterator<'hir> {
    current_id: HirId,
    map: Map<'hir>,
    // Cache the current value of `hir_owner_nodes` to avoid repeatedly calling the same query for
    // the same owner, which will uselessly record many times the same query dependency.
    current_owner_nodes: Option<&'hir OwnerNodes<'hir>>,
}

impl<'hir> ParentHirIterator<'hir> {
    fn new(map: Map<'hir>, current_id: HirId) -> ParentHirIterator<'hir> {
        ParentHirIterator { current_id, map, current_owner_nodes: None }
    }
}

impl<'hir> Iterator for ParentHirIterator<'hir> {
    type Item = HirId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id == CRATE_HIR_ID {
            return None;
        }

        let HirId { owner, local_id } = self.current_id;

        let parent_id = if local_id == ItemLocalId::ZERO {
            // We go from an owner to its parent, so clear the cache.
            self.current_owner_nodes = None;
            self.map.tcx.hir_owner_parent(owner)
        } else {
            let owner_nodes =
                self.current_owner_nodes.get_or_insert_with(|| self.map.tcx.hir_owner_nodes(owner));
            let parent_local_id = owner_nodes.nodes[local_id].parent;
            // HIR indexing should have checked that.
            debug_assert_ne!(parent_local_id, local_id);
            HirId { owner, local_id: parent_local_id }
        };

        debug_assert_ne!(parent_id, self.current_id);

        self.current_id = parent_id;
        Some(parent_id)
    }
}

/// An iterator that walks up the ancestor tree of a given `HirId`.
/// Constructed using `tcx.hir().parent_owner_iter(hir_id)`.
pub struct ParentOwnerIterator<'hir> {
    current_id: HirId,
    map: Map<'hir>,
}

impl<'hir> Iterator for ParentOwnerIterator<'hir> {
    type Item = (OwnerId, OwnerNode<'hir>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id.local_id.index() != 0 {
            self.current_id.local_id = ItemLocalId::ZERO;
            let node = self.map.tcx.hir_owner_node(self.current_id.owner);
            return Some((self.current_id.owner, node));
        }
        if self.current_id == CRATE_HIR_ID {
            return None;
        }

        let parent_id = self.map.def_key(self.current_id.owner.def_id).parent;
        let parent_id = parent_id.map_or(CRATE_OWNER_ID, |local_def_index| {
            let def_id = LocalDefId { local_def_index };
            self.map.tcx.local_def_id_to_hir_id(def_id).owner
        });
        self.current_id = HirId::make_owner(parent_id.def_id);

        let node = self.map.tcx.hir_owner_node(self.current_id.owner);
        Some((self.current_id.owner, node))
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline]
    fn expect_hir_owner_nodes(self, def_id: LocalDefId) -> &'tcx OwnerNodes<'tcx> {
        self.opt_hir_owner_nodes(def_id)
            .unwrap_or_else(|| span_bug!(self.def_span(def_id), "{def_id:?} is not an owner"))
    }

    #[inline]
    pub fn hir_owner_nodes(self, owner_id: OwnerId) -> &'tcx OwnerNodes<'tcx> {
        self.expect_hir_owner_nodes(owner_id.def_id)
    }

    #[inline]
    fn opt_hir_owner_node(self, def_id: LocalDefId) -> Option<OwnerNode<'tcx>> {
        self.opt_hir_owner_nodes(def_id).map(|nodes| nodes.node())
    }

    #[inline]
    pub fn expect_hir_owner_node(self, def_id: LocalDefId) -> OwnerNode<'tcx> {
        self.expect_hir_owner_nodes(def_id).node()
    }

    #[inline]
    pub fn hir_owner_node(self, owner_id: OwnerId) -> OwnerNode<'tcx> {
        self.hir_owner_nodes(owner_id).node()
    }

    /// Retrieves the `hir::Node` corresponding to `id`.
    pub fn hir_node(self, id: HirId) -> Node<'tcx> {
        self.hir_owner_nodes(id.owner).nodes[id.local_id].node
    }

    /// Retrieves the `hir::Node` corresponding to `id`.
    #[inline]
    pub fn hir_node_by_def_id(self, id: LocalDefId) -> Node<'tcx> {
        self.hir_node(self.local_def_id_to_hir_id(id))
    }

    /// Returns `HirId` of the parent HIR node of node with this `hir_id`.
    /// Returns the same `hir_id` if and only if `hir_id == CRATE_HIR_ID`.
    ///
    /// If calling repeatedly and iterating over parents, prefer [`Map::parent_iter`].
    pub fn parent_hir_id(self, hir_id: HirId) -> HirId {
        let HirId { owner, local_id } = hir_id;
        if local_id == ItemLocalId::ZERO {
            self.hir_owner_parent(owner)
        } else {
            let parent_local_id = self.hir_owner_nodes(owner).nodes[local_id].parent;
            // HIR indexing should have checked that.
            debug_assert_ne!(parent_local_id, local_id);
            HirId { owner, local_id: parent_local_id }
        }
    }

    /// Returns parent HIR node of node with this `hir_id`.
    /// Returns HIR node of the same `hir_id` if and only if `hir_id == CRATE_HIR_ID`.
    pub fn parent_hir_node(self, hir_id: HirId) -> Node<'tcx> {
        self.hir_node(self.parent_hir_id(hir_id))
    }
}

impl<'hir> Map<'hir> {
    #[inline]
    pub fn krate(self) -> &'hir Crate<'hir> {
        self.tcx.hir_crate(())
    }

    #[inline]
    pub fn root_module(self) -> &'hir Mod<'hir> {
        match self.tcx.hir_owner_node(CRATE_OWNER_ID) {
            OwnerNode::Crate(item) => item,
            _ => bug!(),
        }
    }

    #[inline]
    pub fn items(self) -> impl Iterator<Item = ItemId> + 'hir {
        self.tcx.hir_crate_items(()).free_items.iter().copied()
    }

    #[inline]
    pub fn module_items(self, module: LocalModDefId) -> impl Iterator<Item = ItemId> + 'hir {
        self.tcx.hir_module_items(module).free_items()
    }

    pub fn def_key(self, def_id: LocalDefId) -> DefKey {
        // Accessing the DefKey is ok, since it is part of DefPathHash.
        self.tcx.definitions_untracked().def_key(def_id)
    }

    pub fn def_path(self, def_id: LocalDefId) -> DefPath {
        // Accessing the DefPath is ok, since it is part of DefPathHash.
        self.tcx.definitions_untracked().def_path(def_id)
    }

    #[inline]
    pub fn def_path_hash(self, def_id: LocalDefId) -> DefPathHash {
        // Accessing the DefPathHash is ok, it is incr. comp. stable.
        self.tcx.definitions_untracked().def_path_hash(def_id)
    }

    pub fn get_if_local(self, id: DefId) -> Option<Node<'hir>> {
        id.as_local().map(|id| self.tcx.hir_node_by_def_id(id))
    }

    pub fn get_generics(self, id: LocalDefId) -> Option<&'hir Generics<'hir>> {
        self.tcx.opt_hir_owner_node(id)?.generics()
    }

    pub fn item(self, id: ItemId) -> &'hir Item<'hir> {
        self.tcx.hir_owner_node(id.owner_id).expect_item()
    }

    pub fn trait_item(self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.tcx.hir_owner_node(id.owner_id).expect_trait_item()
    }

    pub fn impl_item(self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.tcx.hir_owner_node(id.owner_id).expect_impl_item()
    }

    pub fn foreign_item(self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.tcx.hir_owner_node(id.owner_id).expect_foreign_item()
    }

    pub fn body(self, id: BodyId) -> &'hir Body<'hir> {
        self.tcx.hir_owner_nodes(id.hir_id.owner).bodies[&id.hir_id.local_id]
    }

    #[track_caller]
    pub fn fn_decl_by_hir_id(self, hir_id: HirId) -> Option<&'hir FnDecl<'hir>> {
        self.tcx.hir_node(hir_id).fn_decl()
    }

    #[track_caller]
    pub fn fn_sig_by_hir_id(self, hir_id: HirId) -> Option<&'hir FnSig<'hir>> {
        self.tcx.hir_node(hir_id).fn_sig()
    }

    #[track_caller]
    pub fn enclosing_body_owner(self, hir_id: HirId) -> LocalDefId {
        for (_, node) in self.parent_iter(hir_id) {
            if let Some((def_id, _)) = node.associated_body() {
                return def_id;
            }
        }

        bug!("no `enclosing_body_owner` for hir_id `{}`", hir_id);
    }

    /// Returns the `HirId` that corresponds to the definition of
    /// which this is the body of, i.e., a `fn`, `const` or `static`
    /// item (possibly associated), a closure, or a `hir::AnonConst`.
    pub fn body_owner(self, BodyId { hir_id }: BodyId) -> HirId {
        let parent = self.tcx.parent_hir_id(hir_id);
        assert_eq!(self.tcx.hir_node(parent).body_id().unwrap().hir_id, hir_id, "{hir_id:?}");
        parent
    }

    pub fn body_owner_def_id(self, BodyId { hir_id }: BodyId) -> LocalDefId {
        self.tcx.parent_hir_node(hir_id).associated_body().unwrap().0
    }

    /// Given a `LocalDefId`, returns the `BodyId` associated with it,
    /// if the node is a body owner, otherwise returns `None`.
    pub fn maybe_body_owned_by(self, id: LocalDefId) -> Option<&'hir Body<'hir>> {
        Some(self.body(self.tcx.hir_node_by_def_id(id).body_id()?))
    }

    /// Given a body owner's id, returns the `BodyId` associated with it.
    #[track_caller]
    pub fn body_owned_by(self, id: LocalDefId) -> &'hir Body<'hir> {
        self.maybe_body_owned_by(id).unwrap_or_else(|| {
            let hir_id = self.tcx.local_def_id_to_hir_id(id);
            span_bug!(
                self.span(hir_id),
                "body_owned_by: {} has no associated body",
                self.node_to_string(hir_id)
            );
        })
    }

    pub fn body_param_names(self, id: BodyId) -> impl Iterator<Item = Ident> + 'hir {
        self.body(id).params.iter().map(|arg| match arg.pat.kind {
            PatKind::Binding(_, _, ident, _) => ident,
            _ => Ident::empty(),
        })
    }

    /// Returns the `BodyOwnerKind` of this `LocalDefId`.
    ///
    /// Panics if `LocalDefId` does not have an associated body.
    pub fn body_owner_kind(self, def_id: impl Into<DefId>) -> BodyOwnerKind {
        let def_id = def_id.into();
        match self.tcx.def_kind(def_id) {
            DefKind::Const | DefKind::AssocConst | DefKind::AnonConst => {
                BodyOwnerKind::Const { inline: false }
            }
            DefKind::InlineConst => BodyOwnerKind::Const { inline: true },
            DefKind::Ctor(..) | DefKind::Fn | DefKind::AssocFn => BodyOwnerKind::Fn,
            DefKind::Closure | DefKind::SyntheticCoroutineBody => BodyOwnerKind::Closure,
            DefKind::Static { safety: _, mutability, nested: false } => {
                BodyOwnerKind::Static(mutability)
            }
            dk => bug!("{:?} is not a body node: {:?}", def_id, dk),
        }
    }

    /// Returns the `ConstContext` of the body associated with this `LocalDefId`.
    ///
    /// Panics if `LocalDefId` does not have an associated body.
    ///
    /// This should only be used for determining the context of a body, a return
    /// value of `Some` does not always suggest that the owner of the body is `const`,
    /// just that it has to be checked as if it were.
    pub fn body_const_context(self, def_id: impl Into<DefId>) -> Option<ConstContext> {
        let def_id = def_id.into();
        let ccx = match self.body_owner_kind(def_id) {
            BodyOwnerKind::Const { inline } => ConstContext::Const { inline },
            BodyOwnerKind::Static(mutability) => ConstContext::Static(mutability),

            BodyOwnerKind::Fn if self.tcx.is_constructor(def_id) => return None,
            BodyOwnerKind::Fn | BodyOwnerKind::Closure if self.tcx.is_const_fn(def_id) => {
                ConstContext::ConstFn
            }
            BodyOwnerKind::Fn if self.tcx.is_const_default_method(def_id) => ConstContext::ConstFn,
            BodyOwnerKind::Fn | BodyOwnerKind::Closure => return None,
        };

        Some(ccx)
    }

    /// Returns an iterator of the `DefId`s for all body-owners in this
    /// crate. If you would prefer to iterate over the bodies
    /// themselves, you can do `self.hir().krate().body_ids.iter()`.
    #[inline]
    pub fn body_owners(self) -> impl Iterator<Item = LocalDefId> + 'hir {
        self.tcx.hir_crate_items(()).body_owners.iter().copied()
    }

    #[inline]
    pub fn par_body_owners(self, f: impl Fn(LocalDefId) + DynSend + DynSync) {
        par_for_each_in(&self.tcx.hir_crate_items(()).body_owners[..], |&def_id| f(def_id));
    }

    pub fn ty_param_owner(self, def_id: LocalDefId) -> LocalDefId {
        let def_kind = self.tcx.def_kind(def_id);
        match def_kind {
            DefKind::Trait | DefKind::TraitAlias => def_id,
            DefKind::LifetimeParam | DefKind::TyParam | DefKind::ConstParam => {
                self.tcx.local_parent(def_id)
            }
            _ => bug!("ty_param_owner: {:?} is a {:?} not a type parameter", def_id, def_kind),
        }
    }

    pub fn ty_param_name(self, def_id: LocalDefId) -> Symbol {
        let def_kind = self.tcx.def_kind(def_id);
        match def_kind {
            DefKind::Trait | DefKind::TraitAlias => kw::SelfUpper,
            DefKind::LifetimeParam | DefKind::TyParam | DefKind::ConstParam => {
                self.tcx.item_name(def_id.to_def_id())
            }
            _ => bug!("ty_param_name: {:?} is a {:?} not a type parameter", def_id, def_kind),
        }
    }

    pub fn trait_impls(self, trait_did: DefId) -> &'hir [LocalDefId] {
        self.tcx.all_local_trait_impls(()).get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    /// Gets the attributes on the crate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(self) -> &'hir [ast::Attribute] {
        self.attrs(CRATE_HIR_ID)
    }

    pub fn rustc_coherence_is_core(self) -> bool {
        self.krate_attrs().iter().any(|attr| attr.has_name(sym::rustc_coherence_is_core))
    }

    pub fn get_module(self, module: LocalModDefId) -> (&'hir Mod<'hir>, Span, HirId) {
        let hir_id = HirId::make_owner(module.to_local_def_id());
        match self.tcx.hir_owner_node(hir_id.owner) {
            OwnerNode::Item(&Item { span, kind: ItemKind::Mod(m), .. }) => (m, span, hir_id),
            OwnerNode::Crate(item) => (item, item.spans.inner_span, hir_id),
            node => panic!("not a module: {node:?}"),
        }
    }

    /// Walks the contents of the local crate. See also `visit_all_item_likes_in_crate`.
    pub fn walk_toplevel_module<V>(self, visitor: &mut V) -> V::Result
    where
        V: Visitor<'hir>,
    {
        let (top_mod, span, hir_id) = self.get_module(LocalModDefId::CRATE_DEF_ID);
        visitor.visit_mod(top_mod, span, hir_id)
    }

    /// Walks the attributes in a crate.
    pub fn walk_attributes<V>(self, visitor: &mut V) -> V::Result
    where
        V: Visitor<'hir>,
    {
        let krate = self.krate();
        for info in krate.owners.iter() {
            if let MaybeOwner::Owner(info) = info {
                for attrs in info.attrs.map.values() {
                    walk_list!(visitor, visit_attribute, *attrs);
                }
            }
        }
        V::Result::output()
    }

    /// Visits all item-likes in the crate in some deterministic (but unspecified) order. If you
    /// need to process every item-like, and don't care about visiting nested items in a particular
    /// order then this method is the best choice. If you do care about this nesting, you should
    /// use the `tcx.hir().walk_toplevel_module`.
    ///
    /// Note that this function will access HIR for all the item-likes in the crate. If you only
    /// need to access some of them, it is usually better to manually loop on the iterators
    /// provided by `tcx.hir_crate_items(())`.
    ///
    /// Please see the notes in `intravisit.rs` for more information.
    pub fn visit_all_item_likes_in_crate<V>(self, visitor: &mut V) -> V::Result
    where
        V: Visitor<'hir>,
    {
        let krate = self.tcx.hir_crate_items(());
        walk_list!(visitor, visit_item, krate.free_items().map(|id| self.item(id)));
        walk_list!(visitor, visit_trait_item, krate.trait_items().map(|id| self.trait_item(id)));
        walk_list!(visitor, visit_impl_item, krate.impl_items().map(|id| self.impl_item(id)));
        walk_list!(
            visitor,
            visit_foreign_item,
            krate.foreign_items().map(|id| self.foreign_item(id))
        );
        V::Result::output()
    }

    /// This method is the equivalent of `visit_all_item_likes_in_crate` but restricted to
    /// item-likes in a single module.
    pub fn visit_item_likes_in_module<V>(self, module: LocalModDefId, visitor: &mut V) -> V::Result
    where
        V: Visitor<'hir>,
    {
        let module = self.tcx.hir_module_items(module);
        walk_list!(visitor, visit_item, module.free_items().map(|id| self.item(id)));
        walk_list!(visitor, visit_trait_item, module.trait_items().map(|id| self.trait_item(id)));
        walk_list!(visitor, visit_impl_item, module.impl_items().map(|id| self.impl_item(id)));
        walk_list!(
            visitor,
            visit_foreign_item,
            module.foreign_items().map(|id| self.foreign_item(id))
        );
        V::Result::output()
    }

    pub fn for_each_module(self, mut f: impl FnMut(LocalModDefId)) {
        let crate_items = self.tcx.hir_crate_items(());
        for module in crate_items.submodules.iter() {
            f(LocalModDefId::new_unchecked(module.def_id))
        }
    }

    #[inline]
    pub fn par_for_each_module(self, f: impl Fn(LocalModDefId) + DynSend + DynSync) {
        let crate_items = self.tcx.hir_crate_items(());
        par_for_each_in(&crate_items.submodules[..], |module| {
            f(LocalModDefId::new_unchecked(module.def_id))
        })
    }

    #[inline]
    pub fn try_par_for_each_module(
        self,
        f: impl Fn(LocalModDefId) -> Result<(), ErrorGuaranteed> + DynSend + DynSync,
    ) -> Result<(), ErrorGuaranteed> {
        let crate_items = self.tcx.hir_crate_items(());
        try_par_for_each_in(&crate_items.submodules[..], |module| {
            f(LocalModDefId::new_unchecked(module.def_id))
        })
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_id_iter(self, current_id: HirId) -> impl Iterator<Item = HirId> + 'hir {
        ParentHirIterator::new(self, current_id)
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_iter(self, current_id: HirId) -> impl Iterator<Item = (HirId, Node<'hir>)> {
        self.parent_id_iter(current_id).map(move |id| (id, self.tcx.hir_node(id)))
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_owner_iter(self, current_id: HirId) -> ParentOwnerIterator<'hir> {
        ParentOwnerIterator { current_id, map: self }
    }

    /// Checks if the node is left-hand side of an assignment.
    pub fn is_lhs(self, id: HirId) -> bool {
        match self.tcx.parent_hir_node(id) {
            Node::Expr(expr) => match expr.kind {
                ExprKind::Assign(lhs, _rhs, _span) => lhs.hir_id == id,
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether the expression pointed at by `hir_id` belongs to a `const` evaluation context.
    /// Used exclusively for diagnostics, to avoid suggestion function calls.
    pub fn is_inside_const_context(self, hir_id: HirId) -> bool {
        self.body_const_context(self.enclosing_body_owner(hir_id)).is_some()
    }

    /// Retrieves the `HirId` for `id`'s enclosing function *if* the `id` block or return is
    /// in the "tail" position of the function, in other words if it's likely to correspond
    /// to the return type of the function.
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     if x == 1 {
    ///         true  // If `get_fn_id_for_return_block` gets passed the `id` corresponding
    ///     } else {  // to this, it will return `foo`'s `HirId`.
    ///         false
    ///     }
    /// }
    /// ```
    ///
    /// ```compile_fail,E0308
    /// fn foo(x: usize) -> bool {
    ///     loop {
    ///         true  // If `get_fn_id_for_return_block` gets passed the `id` corresponding
    ///     }         // to this, it will return `None`.
    ///     false
    /// }
    /// ```
    pub fn get_fn_id_for_return_block(self, id: HirId) -> Option<HirId> {
        let enclosing_body_owner = self.tcx.local_def_id_to_hir_id(self.enclosing_body_owner(id));

        // Return `None` if the `id` expression is not the returned value of the enclosing body
        let mut iter = [id].into_iter().chain(self.parent_id_iter(id)).peekable();
        while let Some(cur_id) = iter.next() {
            if enclosing_body_owner == cur_id {
                break;
            }

            // A return statement is always the value returned from the enclosing body regardless of
            // what the parent expressions are.
            if let Node::Expr(Expr { kind: ExprKind::Ret(_), .. }) = self.tcx.hir_node(cur_id) {
                break;
            }

            // If the current expression's value doesnt get used as the parent expressions value then return `None`
            if let Some(&parent_id) = iter.peek() {
                match self.tcx.hir_node(parent_id) {
                    // The current node is not the tail expression of the block expression parent expr.
                    Node::Block(Block { expr: Some(e), .. }) if cur_id != e.hir_id => return None,
                    Node::Block(Block { expr: Some(e), .. })
                        if matches!(e.kind, ExprKind::If(_, _, None)) =>
                    {
                        return None;
                    }

                    // The current expression's value does not pass up through these parent expressions
                    Node::Block(Block { expr: None, .. })
                    | Node::Expr(Expr { kind: ExprKind::Loop(..), .. })
                    | Node::LetStmt(..) => return None,

                    _ => {}
                }
            }
        }

        Some(enclosing_body_owner)
    }

    /// Retrieves the `OwnerId` for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the HIR which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent_item(self, hir_id: HirId) -> OwnerId {
        if hir_id.local_id != ItemLocalId::ZERO {
            // If this is a child of a HIR owner, return the owner.
            hir_id.owner
        } else if let Some((def_id, _node)) = self.parent_owner_iter(hir_id).next() {
            def_id
        } else {
            CRATE_OWNER_ID
        }
    }

    /// When on an if expression, a match arm tail expression or a match arm, give back
    /// the enclosing `if` or `match` expression.
    ///
    /// Used by error reporting when there's a type error in an if or match arm caused by the
    /// expression needing to be unit.
    pub fn get_if_cause(self, hir_id: HirId) -> Option<&'hir Expr<'hir>> {
        for (_, node) in self.parent_iter(hir_id) {
            match node {
                Node::Item(_)
                | Node::ForeignItem(_)
                | Node::TraitItem(_)
                | Node::ImplItem(_)
                | Node::Stmt(Stmt { kind: StmtKind::Let(_), .. }) => break,
                Node::Expr(expr @ Expr { kind: ExprKind::If(..) | ExprKind::Match(..), .. }) => {
                    return Some(expr);
                }
                _ => {}
            }
        }
        None
    }

    /// Returns the nearest enclosing scope. A scope is roughly an item or block.
    pub fn get_enclosing_scope(self, hir_id: HirId) -> Option<HirId> {
        for (hir_id, node) in self.parent_iter(hir_id) {
            if let Node::Item(Item {
                kind:
                    ItemKind::Fn(..)
                    | ItemKind::Const(..)
                    | ItemKind::Static(..)
                    | ItemKind::Mod(..)
                    | ItemKind::Enum(..)
                    | ItemKind::Struct(..)
                    | ItemKind::Union(..)
                    | ItemKind::Trait(..)
                    | ItemKind::Impl { .. },
                ..
            })
            | Node::ForeignItem(ForeignItem { kind: ForeignItemKind::Fn(..), .. })
            | Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(..), .. })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(..), .. })
            | Node::Block(_) = node
            {
                return Some(hir_id);
            }
        }
        None
    }

    /// Returns the defining scope for an opaque type definition.
    pub fn get_defining_scope(self, id: HirId) -> HirId {
        let mut scope = id;
        loop {
            scope = self.get_enclosing_scope(scope).unwrap_or(CRATE_HIR_ID);
            if scope == CRATE_HIR_ID || !matches!(self.tcx.hir_node(scope), Node::Block(_)) {
                return scope;
            }
        }
    }

    pub fn get_foreign_abi(self, hir_id: HirId) -> ExternAbi {
        let parent = self.get_parent_item(hir_id);
        if let OwnerNode::Item(Item { kind: ItemKind::ForeignMod { abi, .. }, .. }) =
            self.tcx.hir_owner_node(parent)
        {
            return *abi;
        }
        bug!(
            "expected foreign mod or inlined parent, found {}",
            self.node_to_string(HirId::make_owner(parent.def_id))
        )
    }

    pub fn expect_item(self, id: LocalDefId) -> &'hir Item<'hir> {
        match self.tcx.expect_hir_owner_node(id) {
            OwnerNode::Item(item) => item,
            _ => bug!("expected item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_impl_item(self, id: LocalDefId) -> &'hir ImplItem<'hir> {
        match self.tcx.expect_hir_owner_node(id) {
            OwnerNode::ImplItem(item) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_trait_item(self, id: LocalDefId) -> &'hir TraitItem<'hir> {
        match self.tcx.expect_hir_owner_node(id) {
            OwnerNode::TraitItem(item) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn get_fn_output(self, def_id: LocalDefId) -> Option<&'hir FnRetTy<'hir>> {
        Some(&self.tcx.opt_hir_owner_node(def_id)?.fn_decl()?.output)
    }

    pub fn expect_variant(self, id: HirId) -> &'hir Variant<'hir> {
        match self.tcx.hir_node(id) {
            Node::Variant(variant) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_field(self, id: HirId) -> &'hir FieldDef<'hir> {
        match self.tcx.hir_node(id) {
            Node::Field(field) => field,
            _ => bug!("expected field, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(self, id: OwnerId) -> &'hir ForeignItem<'hir> {
        match self.tcx.hir_owner_node(id) {
            OwnerNode::ForeignItem(item) => item,
            _ => {
                bug!(
                    "expected foreign item, found {}",
                    self.node_to_string(HirId::make_owner(id.def_id))
                )
            }
        }
    }

    #[track_caller]
    pub fn expect_opaque_ty(self, id: LocalDefId) -> &'hir OpaqueTy<'hir> {
        match self.tcx.hir_node_by_def_id(id) {
            Node::OpaqueTy(opaq) => opaq,
            _ => {
                bug!(
                    "expected opaque type definition, found {}",
                    self.node_to_string(self.tcx.local_def_id_to_hir_id(id))
                )
            }
        }
    }

    pub fn expect_expr(self, id: HirId) -> &'hir Expr<'hir> {
        match self.tcx.hir_node(id) {
            Node::Expr(expr) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id)),
        }
    }

    pub fn opt_delegation_sig_id(self, def_id: LocalDefId) -> Option<DefId> {
        self.tcx.opt_hir_owner_node(def_id)?.fn_decl()?.opt_delegation_sig_id()
    }

    #[inline]
    fn opt_ident(self, id: HirId) -> Option<Ident> {
        match self.tcx.hir_node(id) {
            Node::Pat(&Pat { kind: PatKind::Binding(_, _, ident, _), .. }) => Some(ident),
            // A `Ctor` doesn't have an identifier itself, but its parent
            // struct/variant does. Compare with `hir::Map::span`.
            Node::Ctor(..) => match self.tcx.parent_hir_node(id) {
                Node::Item(item) => Some(item.ident),
                Node::Variant(variant) => Some(variant.ident),
                _ => unreachable!(),
            },
            node => node.ident(),
        }
    }

    #[inline]
    pub(super) fn opt_ident_span(self, id: HirId) -> Option<Span> {
        self.opt_ident(id).map(|ident| ident.span)
    }

    #[inline]
    pub fn ident(self, id: HirId) -> Ident {
        self.opt_ident(id).unwrap()
    }

    #[inline]
    pub fn opt_name(self, id: HirId) -> Option<Symbol> {
        self.opt_ident(id).map(|ident| ident.name)
    }

    pub fn name(self, id: HirId) -> Symbol {
        self.opt_name(id).unwrap_or_else(|| bug!("no name for {}", self.node_to_string(id)))
    }

    /// Given a node ID, gets a list of attributes associated with the AST
    /// corresponding to the node-ID.
    pub fn attrs(self, id: HirId) -> &'hir [ast::Attribute] {
        self.tcx.hir_attrs(id.owner).get(id.local_id)
    }

    /// Gets the span of the definition of the specified HIR node.
    /// This is used by `tcx.def_span`.
    pub fn span(self, hir_id: HirId) -> Span {
        fn until_within(outer: Span, end: Span) -> Span {
            if let Some(end) = end.find_ancestor_inside(outer) {
                outer.with_hi(end.hi())
            } else {
                outer
            }
        }

        fn named_span(item_span: Span, ident: Ident, generics: Option<&Generics<'_>>) -> Span {
            if ident.name != kw::Empty {
                let mut span = until_within(item_span, ident.span);
                if let Some(g) = generics
                    && !g.span.is_dummy()
                    && let Some(g_span) = g.span.find_ancestor_inside(item_span)
                {
                    span = span.to(g_span);
                }
                span
            } else {
                item_span
            }
        }

        let span = match self.tcx.hir_node(hir_id) {
            // Function-like.
            Node::Item(Item { kind: ItemKind::Fn(sig, ..), span: outer_span, .. })
            | Node::TraitItem(TraitItem {
                kind: TraitItemKind::Fn(sig, ..),
                span: outer_span,
                ..
            })
            | Node::ImplItem(ImplItem {
                kind: ImplItemKind::Fn(sig, ..), span: outer_span, ..
            })
            | Node::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Fn(sig, ..),
                span: outer_span,
                ..
            }) => {
                // Ensure that the returned span has the item's SyntaxContext, and not the
                // SyntaxContext of the visibility.
                sig.span.find_ancestor_in_same_ctxt(*outer_span).unwrap_or(*outer_span)
            }
            // Impls, including their where clauses.
            Node::Item(Item {
                kind: ItemKind::Impl(Impl { generics, .. }),
                span: outer_span,
                ..
            }) => until_within(*outer_span, generics.where_clause_span),
            // Constants and Statics.
            Node::Item(Item {
                kind: ItemKind::Const(ty, ..) | ItemKind::Static(ty, ..),
                span: outer_span,
                ..
            })
            | Node::TraitItem(TraitItem {
                kind: TraitItemKind::Const(ty, ..),
                span: outer_span,
                ..
            })
            | Node::ImplItem(ImplItem {
                kind: ImplItemKind::Const(ty, ..),
                span: outer_span,
                ..
            })
            | Node::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Static(ty, ..),
                span: outer_span,
                ..
            }) => until_within(*outer_span, ty.span),
            // With generics and bounds.
            Node::Item(Item {
                kind: ItemKind::Trait(_, _, generics, bounds, _),
                span: outer_span,
                ..
            })
            | Node::TraitItem(TraitItem {
                kind: TraitItemKind::Type(bounds, _),
                generics,
                span: outer_span,
                ..
            }) => {
                let end = if let Some(b) = bounds.last() { b.span() } else { generics.span };
                until_within(*outer_span, end)
            }
            // Other cases.
            Node::Item(item) => match &item.kind {
                ItemKind::Use(path, _) => {
                    // Ensure that the returned span has the item's SyntaxContext, and not the
                    // SyntaxContext of the path.
                    path.span.find_ancestor_in_same_ctxt(item.span).unwrap_or(item.span)
                }
                _ => named_span(item.span, item.ident, item.kind.generics()),
            },
            Node::Variant(variant) => named_span(variant.span, variant.ident, None),
            Node::ImplItem(item) => named_span(item.span, item.ident, Some(item.generics)),
            Node::ForeignItem(item) => named_span(item.span, item.ident, None),
            Node::Ctor(_) => return self.span(self.tcx.parent_hir_id(hir_id)),
            Node::Expr(Expr {
                kind: ExprKind::Closure(Closure { fn_decl_span, .. }),
                span,
                ..
            }) => {
                // Ensure that the returned span has the item's SyntaxContext.
                fn_decl_span.find_ancestor_inside(*span).unwrap_or(*span)
            }
            _ => self.span_with_body(hir_id),
        };
        debug_assert_eq!(span.ctxt(), self.span_with_body(hir_id).ctxt());
        span
    }

    /// Like `hir.span()`, but includes the body of items
    /// (instead of just the item header)
    pub fn span_with_body(self, hir_id: HirId) -> Span {
        match self.tcx.hir_node(hir_id) {
            Node::Param(param) => param.span,
            Node::Item(item) => item.span,
            Node::ForeignItem(foreign_item) => foreign_item.span,
            Node::TraitItem(trait_item) => trait_item.span,
            Node::ImplItem(impl_item) => impl_item.span,
            Node::Variant(variant) => variant.span,
            Node::Field(field) => field.span,
            Node::AnonConst(constant) => constant.span,
            Node::ConstBlock(constant) => self.body(constant.body).value.span,
            Node::ConstArg(const_arg) => const_arg.span(),
            Node::Expr(expr) => expr.span,
            Node::ExprField(field) => field.span,
            Node::Stmt(stmt) => stmt.span,
            Node::PathSegment(seg) => {
                let ident_span = seg.ident.span;
                ident_span
                    .with_hi(seg.args.map_or_else(|| ident_span.hi(), |args| args.span_ext.hi()))
            }
            Node::Ty(ty) => ty.span,
            Node::AssocItemConstraint(constraint) => constraint.span,
            Node::TraitRef(tr) => tr.path.span,
            Node::OpaqueTy(op) => op.span,
            Node::Pat(pat) => pat.span,
            Node::PatField(field) => field.span,
            Node::Arm(arm) => arm.span,
            Node::Block(block) => block.span,
            Node::Ctor(..) => self.span_with_body(self.tcx.parent_hir_id(hir_id)),
            Node::Lifetime(lifetime) => lifetime.ident.span,
            Node::GenericParam(param) => param.span,
            Node::Infer(i) => i.span,
            Node::LetStmt(local) => local.span,
            Node::Crate(item) => item.spans.inner_span,
            Node::WherePredicate(pred) => pred.span,
            Node::PreciseCapturingNonLifetimeArg(param) => param.ident.span,
            Node::Synthetic => unreachable!(),
            Node::Err(span) => span,
        }
    }

    pub fn span_if_local(self, id: DefId) -> Option<Span> {
        id.is_local().then(|| self.tcx.def_span(id))
    }

    pub fn res_span(self, res: Res) -> Option<Span> {
        match res {
            Res::Err => None,
            Res::Local(id) => Some(self.span(id)),
            res => self.span_if_local(res.opt_def_id()?),
        }
    }

    /// Get a representation of this `id` for debugging purposes.
    /// NOTE: Do NOT use this in diagnostics!
    pub fn node_to_string(self, id: HirId) -> String {
        hir_id_to_string(self, id)
    }

    /// Returns the HirId of `N` in `struct Foo<const N: usize = { ... }>` when
    /// called with the HirId for the `{ ... }` anon const
    pub fn opt_const_param_default_param_def_id(self, anon_const: HirId) -> Option<LocalDefId> {
        let const_arg = self.tcx.parent_hir_id(anon_const);
        match self.tcx.parent_hir_node(const_arg) {
            Node::GenericParam(GenericParam {
                def_id: param_id,
                kind: GenericParamKind::Const { .. },
                ..
            }) => Some(*param_id),
            _ => None,
        }
    }

    pub fn maybe_get_struct_pattern_shorthand_field(&self, expr: &Expr<'_>) -> Option<Symbol> {
        let local = match expr {
            Expr {
                kind:
                    ExprKind::Path(QPath::Resolved(
                        None,
                        Path {
                            res: def::Res::Local(_), segments: [PathSegment { ident, .. }], ..
                        },
                    )),
                ..
            } => Some(ident),
            _ => None,
        }?;

        match self.tcx.parent_hir_node(expr.hir_id) {
            Node::ExprField(field) => {
                if field.ident.name == local.name && field.is_shorthand {
                    return Some(local.name);
                }
            }
            _ => {}
        }

        None
    }
}

impl<'hir> intravisit::Map<'hir> for Map<'hir> {
    fn hir_node(&self, hir_id: HirId) -> Node<'hir> {
        self.tcx.hir_node(hir_id)
    }

    fn hir_node_by_def_id(&self, def_id: LocalDefId) -> Node<'hir> {
        self.tcx.hir_node_by_def_id(def_id)
    }

    fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        (*self).body(id)
    }

    fn item(&self, id: ItemId) -> &'hir Item<'hir> {
        (*self).item(id)
    }

    fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        (*self).trait_item(id)
    }

    fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        (*self).impl_item(id)
    }

    fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        (*self).foreign_item(id)
    }
}

impl<'tcx> pprust_hir::PpAnn for TyCtxt<'tcx> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        pprust_hir::PpAnn::nested(&(&self.hir() as &dyn intravisit::Map<'_>), state, nested)
    }
}

pub(super) fn crate_hash(tcx: TyCtxt<'_>, _: LocalCrate) -> Svh {
    let krate = tcx.hir_crate(());
    let hir_body_hash = krate.opt_hir_hash.expect("HIR hash missing while computing crate hash");

    let upstream_crates = upstream_crates(tcx);

    let resolutions = tcx.resolutions(());

    // We hash the final, remapped names of all local source files so we
    // don't have to include the path prefix remapping commandline args.
    // If we included the full mapping in the SVH, we could only have
    // reproducible builds by compiling from the same directory. So we just
    // hash the result of the mapping instead of the mapping itself.
    let mut source_file_names: Vec<_> = tcx
        .sess
        .source_map()
        .files()
        .iter()
        .filter(|source_file| source_file.cnum == LOCAL_CRATE)
        .map(|source_file| source_file.stable_id)
        .collect();

    source_file_names.sort_unstable();

    // We have to take care of debugger visualizers explicitly. The HIR (and
    // thus `hir_body_hash`) contains the #[debugger_visualizer] attributes but
    // these attributes only store the file path to the visualizer file, not
    // their content. Yet that content is exported into crate metadata, so any
    // changes to it need to be reflected in the crate hash.
    let debugger_visualizers: Vec<_> = tcx
        .debugger_visualizers(LOCAL_CRATE)
        .iter()
        // We ignore the path to the visualizer file since it's not going to be
        // encoded in crate metadata and we already hash the full contents of
        // the file.
        .map(DebuggerVisualizerFile::path_erased)
        .collect();

    let crate_hash: Fingerprint = tcx.with_stable_hashing_context(|mut hcx| {
        let mut stable_hasher = StableHasher::new();
        hir_body_hash.hash_stable(&mut hcx, &mut stable_hasher);
        upstream_crates.hash_stable(&mut hcx, &mut stable_hasher);
        source_file_names.hash_stable(&mut hcx, &mut stable_hasher);
        debugger_visualizers.hash_stable(&mut hcx, &mut stable_hasher);
        if tcx.sess.opts.incremental.is_some() {
            let definitions = tcx.untracked().definitions.freeze();
            let mut owner_spans: Vec<_> = krate
                .owners
                .iter_enumerated()
                .filter_map(|(def_id, info)| {
                    let _ = info.as_owner()?;
                    let def_path_hash = definitions.def_path_hash(def_id);
                    let span = tcx.source_span(def_id);
                    debug_assert_eq!(span.parent(), None);
                    Some((def_path_hash, span))
                })
                .collect();
            owner_spans.sort_unstable_by_key(|bn| bn.0);
            owner_spans.hash_stable(&mut hcx, &mut stable_hasher);
        }
        tcx.sess.opts.dep_tracking_hash(true).hash_stable(&mut hcx, &mut stable_hasher);
        tcx.stable_crate_id(LOCAL_CRATE).hash_stable(&mut hcx, &mut stable_hasher);
        // Hash visibility information since it does not appear in HIR.
        // FIXME: Figure out how to remove `visibilities_for_hashing` by hashing visibilities on
        // the fly in the resolver, storing only their accumulated hash in `ResolverGlobalCtxt`,
        // and combining it with other hashes here.
        resolutions.visibilities_for_hashing.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    });

    Svh::new(crate_hash)
}

fn upstream_crates(tcx: TyCtxt<'_>) -> Vec<(StableCrateId, Svh)> {
    let mut upstream_crates: Vec<_> = tcx
        .crates(())
        .iter()
        .map(|&cnum| {
            let stable_crate_id = tcx.stable_crate_id(cnum);
            let hash = tcx.crate_hash(cnum);
            (stable_crate_id, hash)
        })
        .collect();
    upstream_crates.sort_unstable_by_key(|&(stable_crate_id, _)| stable_crate_id);
    upstream_crates
}

fn hir_id_to_string(map: Map<'_>, id: HirId) -> String {
    let path_str = |def_id: LocalDefId| map.tcx.def_path_str(def_id);

    let span_str = || map.tcx.sess.source_map().span_to_snippet(map.span(id)).unwrap_or_default();
    let node_str = |prefix| format!("{id} ({prefix} `{}`)", span_str());

    match map.tcx.hir_node(id) {
        Node::Item(item) => {
            let item_str = match item.kind {
                ItemKind::ExternCrate(..) => "extern crate",
                ItemKind::Use(..) => "use",
                ItemKind::Static(..) => "static",
                ItemKind::Const(..) => "const",
                ItemKind::Fn(..) => "fn",
                ItemKind::Macro(..) => "macro",
                ItemKind::Mod(..) => "mod",
                ItemKind::ForeignMod { .. } => "foreign mod",
                ItemKind::GlobalAsm(..) => "global asm",
                ItemKind::TyAlias(..) => "ty",
                ItemKind::Enum(..) => "enum",
                ItemKind::Struct(..) => "struct",
                ItemKind::Union(..) => "union",
                ItemKind::Trait(..) => "trait",
                ItemKind::TraitAlias(..) => "trait alias",
                ItemKind::Impl { .. } => "impl",
            };
            format!("{id} ({item_str} {})", path_str(item.owner_id.def_id))
        }
        Node::ForeignItem(item) => {
            format!("{id} (foreign item {})", path_str(item.owner_id.def_id))
        }
        Node::ImplItem(ii) => {
            let kind = match ii.kind {
                ImplItemKind::Const(..) => "associated constant",
                ImplItemKind::Fn(fn_sig, _) => match fn_sig.decl.implicit_self {
                    ImplicitSelfKind::None => "associated function",
                    _ => "method",
                },
                ImplItemKind::Type(_) => "associated type",
            };
            format!("{id} ({kind} `{}` in {})", ii.ident, path_str(ii.owner_id.def_id))
        }
        Node::TraitItem(ti) => {
            let kind = match ti.kind {
                TraitItemKind::Const(..) => "associated constant",
                TraitItemKind::Fn(fn_sig, _) => match fn_sig.decl.implicit_self {
                    ImplicitSelfKind::None => "associated function",
                    _ => "trait method",
                },
                TraitItemKind::Type(..) => "associated type",
            };

            format!("{id} ({kind} `{}` in {})", ti.ident, path_str(ti.owner_id.def_id))
        }
        Node::Variant(variant) => {
            format!("{id} (variant `{}` in {})", variant.ident, path_str(variant.def_id))
        }
        Node::Field(field) => {
            format!("{id} (field `{}` in {})", field.ident, path_str(field.def_id))
        }
        Node::AnonConst(_) => node_str("const"),
        Node::ConstBlock(_) => node_str("const"),
        Node::ConstArg(_) => node_str("const"),
        Node::Expr(_) => node_str("expr"),
        Node::ExprField(_) => node_str("expr field"),
        Node::Stmt(_) => node_str("stmt"),
        Node::PathSegment(_) => node_str("path segment"),
        Node::Ty(_) => node_str("type"),
        Node::AssocItemConstraint(_) => node_str("assoc item constraint"),
        Node::TraitRef(_) => node_str("trait ref"),
        Node::OpaqueTy(_) => node_str("opaque type"),
        Node::Pat(_) => node_str("pat"),
        Node::PatField(_) => node_str("pattern field"),
        Node::Param(_) => node_str("param"),
        Node::Arm(_) => node_str("arm"),
        Node::Block(_) => node_str("block"),
        Node::Infer(_) => node_str("infer"),
        Node::LetStmt(_) => node_str("local"),
        Node::Ctor(ctor) => format!(
            "{id} (ctor {})",
            ctor.ctor_def_id().map_or("<missing path>".into(), |def_id| path_str(def_id)),
        ),
        Node::Lifetime(_) => node_str("lifetime"),
        Node::GenericParam(param) => {
            format!("{id} (generic_param {})", path_str(param.def_id))
        }
        Node::Crate(..) => String::from("(root_crate)"),
        Node::WherePredicate(_) => node_str("where predicate"),
        Node::Synthetic => unreachable!(),
        Node::Err(_) => node_str("error"),
        Node::PreciseCapturingNonLifetimeArg(_param) => node_str("parameter"),
    }
}

pub(super) fn hir_module_items(tcx: TyCtxt<'_>, module_id: LocalModDefId) -> ModuleItems {
    let mut collector = ItemCollector::new(tcx, false);

    let (hir_mod, span, hir_id) = tcx.hir().get_module(module_id);
    collector.visit_mod(hir_mod, span, hir_id);

    let ItemCollector {
        submodules,
        items,
        trait_items,
        impl_items,
        foreign_items,
        body_owners,
        opaques,
        ..
    } = collector;
    ModuleItems {
        submodules: submodules.into_boxed_slice(),
        free_items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
        opaques: opaques.into_boxed_slice(),
    }
}

pub(crate) fn hir_crate_items(tcx: TyCtxt<'_>, _: ()) -> ModuleItems {
    let mut collector = ItemCollector::new(tcx, true);

    // A "crate collector" and "module collector" start at a
    // module item (the former starts at the crate root) but only
    // the former needs to collect it. ItemCollector does not do this for us.
    collector.submodules.push(CRATE_OWNER_ID);
    tcx.hir().walk_toplevel_module(&mut collector);

    let ItemCollector {
        submodules,
        items,
        trait_items,
        impl_items,
        foreign_items,
        body_owners,
        opaques,
        ..
    } = collector;

    ModuleItems {
        submodules: submodules.into_boxed_slice(),
        free_items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
        opaques: opaques.into_boxed_slice(),
    }
}

struct ItemCollector<'tcx> {
    // When true, it collects all items in the create,
    // otherwise it collects items in some module.
    crate_collector: bool,
    tcx: TyCtxt<'tcx>,
    submodules: Vec<OwnerId>,
    items: Vec<ItemId>,
    trait_items: Vec<TraitItemId>,
    impl_items: Vec<ImplItemId>,
    foreign_items: Vec<ForeignItemId>,
    body_owners: Vec<LocalDefId>,
    opaques: Vec<LocalDefId>,
}

impl<'tcx> ItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, crate_collector: bool) -> ItemCollector<'tcx> {
        ItemCollector {
            crate_collector,
            tcx,
            submodules: Vec::default(),
            items: Vec::default(),
            trait_items: Vec::default(),
            impl_items: Vec::default(),
            foreign_items: Vec::default(),
            body_owners: Vec::default(),
            opaques: Vec::default(),
        }
    }
}

impl<'hir> Visitor<'hir> for ItemCollector<'hir> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'hir Item<'hir>) {
        if Node::Item(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.items.push(item.item_id());

        // Items that are modules are handled here instead of in visit_mod.
        if let ItemKind::Mod(module) = &item.kind {
            self.submodules.push(item.owner_id);
            // A module collector does not recurse inside nested modules.
            if self.crate_collector {
                intravisit::walk_mod(self, module, item.hir_id());
            }
        } else {
            intravisit::walk_item(self, item)
        }
    }

    fn visit_foreign_item(&mut self, item: &'hir ForeignItem<'hir>) {
        self.foreign_items.push(item.foreign_item_id());
        intravisit::walk_foreign_item(self, item)
    }

    fn visit_anon_const(&mut self, c: &'hir AnonConst) {
        self.body_owners.push(c.def_id);
        intravisit::walk_anon_const(self, c)
    }

    fn visit_inline_const(&mut self, c: &'hir ConstBlock) {
        self.body_owners.push(c.def_id);
        intravisit::walk_inline_const(self, c)
    }

    fn visit_opaque_ty(&mut self, o: &'hir OpaqueTy<'hir>) {
        self.opaques.push(o.def_id);
        intravisit::walk_opaque_ty(self, o)
    }

    fn visit_expr(&mut self, ex: &'hir Expr<'hir>) {
        if let ExprKind::Closure(closure) = ex.kind {
            self.body_owners.push(closure.def_id);
        }
        intravisit::walk_expr(self, ex)
    }

    fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
        if Node::TraitItem(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.trait_items.push(item.trait_item_id());
        intravisit::walk_trait_item(self, item)
    }

    fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
        if Node::ImplItem(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.impl_items.push(item.impl_item_id());
        intravisit::walk_impl_item(self, item)
    }
}
