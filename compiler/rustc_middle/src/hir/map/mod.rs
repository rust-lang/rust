use crate::hir::{ModuleItems, Owner};
use crate::ty::{DefIdTree, TyCtxt};
use rustc_ast as ast;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{par_for_each_in, Send, Sync};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::*;
use rustc_index::vec::Idx;
use rustc_middle::hir::nested_filter;
use rustc_span::def_id::StableCrateId;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

#[inline]
pub fn associated_body(node: Node<'_>) -> Option<(LocalDefId, BodyId)> {
    match node {
        Node::Item(Item {
            owner_id,
            kind: ItemKind::Const(_, body) | ItemKind::Static(.., body) | ItemKind::Fn(.., body),
            ..
        })
        | Node::TraitItem(TraitItem {
            owner_id,
            kind:
                TraitItemKind::Const(_, Some(body)) | TraitItemKind::Fn(_, TraitFn::Provided(body)),
            ..
        })
        | Node::ImplItem(ImplItem {
            owner_id,
            kind: ImplItemKind::Const(_, body) | ImplItemKind::Fn(_, body),
            ..
        }) => Some((owner_id.def_id, *body)),

        Node::Expr(Expr { kind: ExprKind::Closure(Closure { def_id, body, .. }), .. }) => {
            Some((*def_id, *body))
        }

        Node::AnonConst(constant) => Some((constant.def_id, constant.body)),

        _ => None,
    }
}

fn is_body_owner(node: Node<'_>, hir_id: HirId) -> bool {
    match associated_body(node) {
        Some((_, b)) => b.hir_id == hir_id,
        None => false,
    }
}

#[derive(Copy, Clone)]
pub struct Map<'hir> {
    pub(super) tcx: TyCtxt<'hir>,
}

/// An iterator that walks up the ancestor tree of a given `HirId`.
/// Constructed using `tcx.hir().parent_iter(hir_id)`.
pub struct ParentHirIterator<'hir> {
    current_id: HirId,
    map: Map<'hir>,
}

impl<'hir> Iterator for ParentHirIterator<'hir> {
    type Item = HirId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id == CRATE_HIR_ID {
            return None;
        }
        loop {
            // There are nodes that do not have entries, so we need to skip them.
            let parent_id = self.map.parent_id(self.current_id);

            if parent_id == self.current_id {
                self.current_id = CRATE_HIR_ID;
                return None;
            }

            self.current_id = parent_id;
            return Some(parent_id);
        }
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
            self.current_id.local_id = ItemLocalId::new(0);
            if let Some(node) = self.map.tcx.hir_owner(self.current_id.owner) {
                return Some((self.current_id.owner, node.node));
            }
        }
        if self.current_id == CRATE_HIR_ID {
            return None;
        }
        loop {
            // There are nodes that do not have entries, so we need to skip them.
            let parent_id = self.map.def_key(self.current_id.owner.def_id).parent;

            let parent_id = parent_id.map_or(CRATE_OWNER_ID, |local_def_index| {
                let def_id = LocalDefId { local_def_index };
                self.map.local_def_id_to_hir_id(def_id).owner
            });
            self.current_id = HirId::make_owner(parent_id.def_id);

            // If this `HirId` doesn't have an entry, skip it and look for its `parent_id`.
            if let Some(node) = self.map.tcx.hir_owner(self.current_id.owner) {
                return Some((self.current_id.owner, node.node));
            }
        }
    }
}

impl<'hir> Map<'hir> {
    #[inline]
    pub fn krate(self) -> &'hir Crate<'hir> {
        self.tcx.hir_crate(())
    }

    #[inline]
    pub fn root_module(self) -> &'hir Mod<'hir> {
        match self.tcx.hir_owner(CRATE_OWNER_ID).map(|o| o.node) {
            Some(OwnerNode::Crate(item)) => item,
            _ => bug!(),
        }
    }

    #[inline]
    pub fn items(self) -> impl Iterator<Item = ItemId> + 'hir {
        self.tcx.hir_crate_items(()).items.iter().copied()
    }

    #[inline]
    pub fn module_items(self, module: LocalDefId) -> impl Iterator<Item = ItemId> + 'hir {
        self.tcx.hir_module_items(module).items()
    }

    #[inline]
    pub fn par_for_each_item(self, f: impl Fn(ItemId) + Sync + Send) {
        par_for_each_in(&self.tcx.hir_crate_items(()).items[..], |id| f(*id));
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

    #[inline]
    pub fn local_def_id_to_hir_id(self, def_id: LocalDefId) -> HirId {
        self.tcx.local_def_id_to_hir_id(def_id)
    }

    /// Do not call this function directly. The query should be called.
    pub(super) fn opt_def_kind(self, local_def_id: LocalDefId) -> Option<DefKind> {
        let hir_id = self.local_def_id_to_hir_id(local_def_id);
        let def_kind = match self.find(hir_id)? {
            Node::Item(item) => match item.kind {
                ItemKind::Static(_, mt, _) => DefKind::Static(mt),
                ItemKind::Const(..) => DefKind::Const,
                ItemKind::Fn(..) => DefKind::Fn,
                ItemKind::Macro(_, macro_kind) => DefKind::Macro(macro_kind),
                ItemKind::Mod(..) => DefKind::Mod,
                ItemKind::OpaqueTy(ref opaque) => {
                    if opaque.in_trait {
                        DefKind::ImplTraitPlaceholder
                    } else {
                        DefKind::OpaqueTy
                    }
                }
                ItemKind::TyAlias(..) => DefKind::TyAlias,
                ItemKind::Enum(..) => DefKind::Enum,
                ItemKind::Struct(..) => DefKind::Struct,
                ItemKind::Union(..) => DefKind::Union,
                ItemKind::Trait(..) => DefKind::Trait,
                ItemKind::TraitAlias(..) => DefKind::TraitAlias,
                ItemKind::ExternCrate(_) => DefKind::ExternCrate,
                ItemKind::Use(..) => DefKind::Use,
                ItemKind::ForeignMod { .. } => DefKind::ForeignMod,
                ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
                ItemKind::Impl { .. } => DefKind::Impl,
            },
            Node::ForeignItem(item) => match item.kind {
                ForeignItemKind::Fn(..) => DefKind::Fn,
                ForeignItemKind::Static(_, mt) => DefKind::Static(mt),
                ForeignItemKind::Type => DefKind::ForeignTy,
            },
            Node::TraitItem(item) => match item.kind {
                TraitItemKind::Const(..) => DefKind::AssocConst,
                TraitItemKind::Fn(..) => DefKind::AssocFn,
                TraitItemKind::Type(..) => DefKind::AssocTy,
            },
            Node::ImplItem(item) => match item.kind {
                ImplItemKind::Const(..) => DefKind::AssocConst,
                ImplItemKind::Fn(..) => DefKind::AssocFn,
                ImplItemKind::Type(..) => DefKind::AssocTy,
            },
            Node::Variant(_) => DefKind::Variant,
            Node::Ctor(variant_data) => {
                let ctor_of = match self.find_parent(hir_id) {
                    Some(Node::Item(..)) => def::CtorOf::Struct,
                    Some(Node::Variant(..)) => def::CtorOf::Variant,
                    _ => unreachable!(),
                };
                match variant_data.ctor_kind() {
                    Some(kind) => DefKind::Ctor(ctor_of, kind),
                    None => bug!("constructor node without a constructor"),
                }
            }
            Node::AnonConst(_) => {
                let inline = match self.find_parent(hir_id) {
                    Some(Node::Expr(&Expr {
                        kind: ExprKind::ConstBlock(ref anon_const), ..
                    })) if anon_const.hir_id == hir_id => true,
                    _ => false,
                };
                if inline { DefKind::InlineConst } else { DefKind::AnonConst }
            }
            Node::Field(_) => DefKind::Field,
            Node::Expr(expr) => match expr.kind {
                ExprKind::Closure(Closure { movability: None, .. }) => DefKind::Closure,
                ExprKind::Closure(Closure { movability: Some(_), .. }) => DefKind::Generator,
                _ => bug!("def_kind: unsupported node: {}", self.node_to_string(hir_id)),
            },
            Node::GenericParam(param) => match param.kind {
                GenericParamKind::Lifetime { .. } => DefKind::LifetimeParam,
                GenericParamKind::Type { .. } => DefKind::TyParam,
                GenericParamKind::Const { .. } => DefKind::ConstParam,
            },
            Node::Crate(_) => DefKind::Mod,
            Node::Stmt(_)
            | Node::PathSegment(_)
            | Node::Ty(_)
            | Node::TypeBinding(_)
            | Node::Infer(_)
            | Node::TraitRef(_)
            | Node::Pat(_)
            | Node::PatField(_)
            | Node::ExprField(_)
            | Node::Local(_)
            | Node::Param(_)
            | Node::Arm(_)
            | Node::Lifetime(_)
            | Node::Block(_) => return None,
        };
        Some(def_kind)
    }

    /// Finds the id of the parent node to this one.
    ///
    /// If calling repeatedly and iterating over parents, prefer [`Map::parent_iter`].
    pub fn opt_parent_id(self, id: HirId) -> Option<HirId> {
        if id.local_id == ItemLocalId::from_u32(0) {
            Some(self.tcx.hir_owner_parent(id.owner))
        } else {
            let owner = self.tcx.hir_owner_nodes(id.owner).as_owner()?;
            let node = owner.nodes[id.local_id].as_ref()?;
            let hir_id = HirId { owner: id.owner, local_id: node.parent };
            // HIR indexing should have checked that.
            debug_assert_ne!(id.local_id, node.parent);
            Some(hir_id)
        }
    }

    #[track_caller]
    pub fn parent_id(self, hir_id: HirId) -> HirId {
        self.opt_parent_id(hir_id)
            .unwrap_or_else(|| bug!("No parent for node {:?}", self.node_to_string(hir_id)))
    }

    pub fn get_parent(self, hir_id: HirId) -> Node<'hir> {
        self.get(self.parent_id(hir_id))
    }

    pub fn find_parent(self, hir_id: HirId) -> Option<Node<'hir>> {
        self.find(self.opt_parent_id(hir_id)?)
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    pub fn find(self, id: HirId) -> Option<Node<'hir>> {
        if id.local_id == ItemLocalId::from_u32(0) {
            let owner = self.tcx.hir_owner(id.owner)?;
            Some(owner.node.into())
        } else {
            let owner = self.tcx.hir_owner_nodes(id.owner).as_owner()?;
            let node = owner.nodes[id.local_id].as_ref()?;
            Some(node.node)
        }
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    #[inline]
    pub fn find_by_def_id(self, id: LocalDefId) -> Option<Node<'hir>> {
        self.find(self.local_def_id_to_hir_id(id))
    }

    /// Retrieves the `Node` corresponding to `id`, panicking if it cannot be found.
    #[track_caller]
    pub fn get(self, id: HirId) -> Node<'hir> {
        self.find(id).unwrap_or_else(|| bug!("couldn't find hir id {} in the HIR map", id))
    }

    /// Retrieves the `Node` corresponding to `id`, panicking if it cannot be found.
    #[inline]
    #[track_caller]
    pub fn get_by_def_id(self, id: LocalDefId) -> Node<'hir> {
        self.find_by_def_id(id).unwrap_or_else(|| bug!("couldn't find {:?} in the HIR map", id))
    }

    pub fn get_if_local(self, id: DefId) -> Option<Node<'hir>> {
        id.as_local().and_then(|id| self.find(self.local_def_id_to_hir_id(id)))
    }

    pub fn get_generics(self, id: LocalDefId) -> Option<&'hir Generics<'hir>> {
        let node = self.tcx.hir_owner(OwnerId { def_id: id })?;
        node.node.generics()
    }

    pub fn owner(self, id: OwnerId) -> OwnerNode<'hir> {
        self.tcx.hir_owner(id).unwrap_or_else(|| bug!("expected owner for {:?}", id)).node
    }

    pub fn item(self, id: ItemId) -> &'hir Item<'hir> {
        self.tcx.hir_owner(id.owner_id).unwrap().node.expect_item()
    }

    pub fn trait_item(self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.tcx.hir_owner(id.owner_id).unwrap().node.expect_trait_item()
    }

    pub fn impl_item(self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.tcx.hir_owner(id.owner_id).unwrap().node.expect_impl_item()
    }

    pub fn foreign_item(self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.tcx.hir_owner(id.owner_id).unwrap().node.expect_foreign_item()
    }

    pub fn body(self, id: BodyId) -> &'hir Body<'hir> {
        self.tcx.hir_owner_nodes(id.hir_id.owner).unwrap().bodies[&id.hir_id.local_id]
    }

    #[track_caller]
    pub fn fn_decl_by_hir_id(self, hir_id: HirId) -> Option<&'hir FnDecl<'hir>> {
        if let Some(node) = self.find(hir_id) {
            node.fn_decl()
        } else {
            bug!("no node for hir_id `{}`", hir_id)
        }
    }

    #[track_caller]
    pub fn fn_sig_by_hir_id(self, hir_id: HirId) -> Option<&'hir FnSig<'hir>> {
        if let Some(node) = self.find(hir_id) {
            node.fn_sig()
        } else {
            bug!("no node for hir_id `{}`", hir_id)
        }
    }

    #[track_caller]
    pub fn enclosing_body_owner(self, hir_id: HirId) -> LocalDefId {
        for (_, node) in self.parent_iter(hir_id) {
            if let Some((def_id, _)) = associated_body(node) {
                return def_id;
            }
        }

        bug!("no `enclosing_body_owner` for hir_id `{}`", hir_id);
    }

    /// Returns the `HirId` that corresponds to the definition of
    /// which this is the body of, i.e., a `fn`, `const` or `static`
    /// item (possibly associated), a closure, or a `hir::AnonConst`.
    pub fn body_owner(self, BodyId { hir_id }: BodyId) -> HirId {
        let parent = self.parent_id(hir_id);
        assert!(self.find(parent).map_or(false, |n| is_body_owner(n, hir_id)), "{hir_id:?}");
        parent
    }

    pub fn body_owner_def_id(self, BodyId { hir_id }: BodyId) -> LocalDefId {
        let parent = self.parent_id(hir_id);
        associated_body(self.get(parent)).unwrap().0
    }

    /// Given a `LocalDefId`, returns the `BodyId` associated with it,
    /// if the node is a body owner, otherwise returns `None`.
    pub fn maybe_body_owned_by(self, id: LocalDefId) -> Option<BodyId> {
        let node = self.find_by_def_id(id)?;
        let (_, body_id) = associated_body(node)?;
        Some(body_id)
    }

    /// Given a body owner's id, returns the `BodyId` associated with it.
    #[track_caller]
    pub fn body_owned_by(self, id: LocalDefId) -> BodyId {
        self.maybe_body_owned_by(id).unwrap_or_else(|| {
            let hir_id = self.local_def_id_to_hir_id(id);
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
    pub fn body_owner_kind(self, def_id: LocalDefId) -> BodyOwnerKind {
        match self.tcx.def_kind(def_id) {
            DefKind::Const | DefKind::AssocConst | DefKind::InlineConst | DefKind::AnonConst => {
                BodyOwnerKind::Const
            }
            DefKind::Ctor(..) | DefKind::Fn | DefKind::AssocFn => BodyOwnerKind::Fn,
            DefKind::Closure | DefKind::Generator => BodyOwnerKind::Closure,
            DefKind::Static(mt) => BodyOwnerKind::Static(mt),
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
    pub fn body_const_context(self, def_id: LocalDefId) -> Option<ConstContext> {
        let ccx = match self.body_owner_kind(def_id) {
            BodyOwnerKind::Const => ConstContext::Const,
            BodyOwnerKind::Static(mt) => ConstContext::Static(mt),

            BodyOwnerKind::Fn if self.tcx.is_constructor(def_id.to_def_id()) => return None,
            BodyOwnerKind::Fn | BodyOwnerKind::Closure
                if self.tcx.is_const_fn_raw(def_id.to_def_id()) =>
            {
                ConstContext::ConstFn
            }
            BodyOwnerKind::Fn if self.tcx.is_const_default_method(def_id.to_def_id()) => {
                ConstContext::ConstFn
            }
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
    pub fn par_body_owners(self, f: impl Fn(LocalDefId) + Sync + Send) {
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

    pub fn get_module(self, module: LocalDefId) -> (&'hir Mod<'hir>, Span, HirId) {
        let hir_id = HirId::make_owner(module);
        match self.tcx.hir_owner(hir_id.owner).map(|o| o.node) {
            Some(OwnerNode::Item(&Item { span, kind: ItemKind::Mod(ref m), .. })) => {
                (m, span, hir_id)
            }
            Some(OwnerNode::Crate(item)) => (item, item.spans.inner_span, hir_id),
            node => panic!("not a module: {:?}", node),
        }
    }

    /// Walks the contents of the local crate. See also `visit_all_item_likes_in_crate`.
    pub fn walk_toplevel_module(self, visitor: &mut impl Visitor<'hir>) {
        let (top_mod, span, hir_id) = self.get_module(CRATE_DEF_ID);
        visitor.visit_mod(top_mod, span, hir_id);
    }

    /// Walks the attributes in a crate.
    pub fn walk_attributes(self, visitor: &mut impl Visitor<'hir>) {
        let krate = self.krate();
        for info in krate.owners.iter() {
            if let MaybeOwner::Owner(info) = info {
                for attrs in info.attrs.map.values() {
                    for a in *attrs {
                        visitor.visit_attribute(a)
                    }
                }
            }
        }
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
    pub fn visit_all_item_likes_in_crate<V>(self, visitor: &mut V)
    where
        V: Visitor<'hir>,
    {
        let krate = self.tcx.hir_crate_items(());

        for id in krate.items() {
            visitor.visit_item(self.item(id));
        }

        for id in krate.trait_items() {
            visitor.visit_trait_item(self.trait_item(id));
        }

        for id in krate.impl_items() {
            visitor.visit_impl_item(self.impl_item(id));
        }

        for id in krate.foreign_items() {
            visitor.visit_foreign_item(self.foreign_item(id));
        }
    }

    /// This method is the equivalent of `visit_all_item_likes_in_crate` but restricted to
    /// item-likes in a single module.
    pub fn visit_item_likes_in_module<V>(self, module: LocalDefId, visitor: &mut V)
    where
        V: Visitor<'hir>,
    {
        let module = self.tcx.hir_module_items(module);

        for id in module.items() {
            visitor.visit_item(self.item(id));
        }

        for id in module.trait_items() {
            visitor.visit_trait_item(self.trait_item(id));
        }

        for id in module.impl_items() {
            visitor.visit_impl_item(self.impl_item(id));
        }

        for id in module.foreign_items() {
            visitor.visit_foreign_item(self.foreign_item(id));
        }
    }

    pub fn for_each_module(self, mut f: impl FnMut(LocalDefId)) {
        let crate_items = self.tcx.hir_crate_items(());
        for module in crate_items.submodules.iter() {
            f(module.def_id)
        }
    }

    #[inline]
    pub fn par_for_each_module(self, f: impl Fn(LocalDefId) + Sync + Send) {
        let crate_items = self.tcx.hir_crate_items(());
        par_for_each_in(&crate_items.submodules[..], |module| f(module.def_id))
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_id_iter(self, current_id: HirId) -> impl Iterator<Item = HirId> + 'hir {
        ParentHirIterator { current_id, map: self }
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_iter(self, current_id: HirId) -> impl Iterator<Item = (HirId, Node<'hir>)> {
        self.parent_id_iter(current_id).filter_map(move |id| Some((id, self.find(id)?)))
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `parent_id`.
    #[inline]
    pub fn parent_owner_iter(self, current_id: HirId) -> ParentOwnerIterator<'hir> {
        ParentOwnerIterator { current_id, map: self }
    }

    /// Checks if the node is left-hand side of an assignment.
    pub fn is_lhs(self, id: HirId) -> bool {
        match self.find_parent(id) {
            Some(Node::Expr(expr)) => match expr.kind {
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
    /// ```compile_fail,E0308
    /// fn foo(x: usize) -> bool {
    ///     loop {
    ///         true  // If `get_return_block` gets passed the `id` corresponding
    ///     }         // to this, it will return `None`.
    ///     false
    /// }
    /// ```
    pub fn get_return_block(self, id: HirId) -> Option<HirId> {
        let mut iter = self.parent_iter(id).peekable();
        let mut ignore_tail = false;
        if let Some(Node::Expr(Expr { kind: ExprKind::Ret(_), .. })) = self.find(id) {
            // When dealing with `return` statements, we don't care about climbing only tail
            // expressions.
            ignore_tail = true;
        }
        while let Some((hir_id, node)) = iter.next() {
            if let (Some((_, next_node)), false) = (iter.peek(), ignore_tail) {
                match next_node {
                    Node::Block(Block { expr: None, .. }) => return None,
                    // The current node is not the tail expression of its parent.
                    Node::Block(Block { expr: Some(e), .. }) if hir_id != e.hir_id => return None,
                    _ => {}
                }
            }
            match node {
                Node::Item(_)
                | Node::ForeignItem(_)
                | Node::TraitItem(_)
                | Node::Expr(Expr { kind: ExprKind::Closure { .. }, .. })
                | Node::ImplItem(_) => return Some(hir_id),
                // Ignore `return`s on the first iteration
                Node::Expr(Expr { kind: ExprKind::Loop(..) | ExprKind::Ret(..), .. })
                | Node::Local(_) => {
                    return None;
                }
                _ => {}
            }
        }
        None
    }

    /// Retrieves the `OwnerId` for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the HIR which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent_item(self, hir_id: HirId) -> OwnerId {
        if let Some((def_id, _node)) = self.parent_owner_iter(hir_id).next() {
            def_id
        } else {
            CRATE_OWNER_ID
        }
    }

    /// Returns the `OwnerId` of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub(super) fn get_module_parent_node(self, hir_id: HirId) -> OwnerId {
        for (def_id, node) in self.parent_owner_iter(hir_id) {
            if let OwnerNode::Item(&Item { kind: ItemKind::Mod(_), .. }) = node {
                return def_id;
            }
        }
        CRATE_OWNER_ID
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
                | Node::Stmt(Stmt { kind: StmtKind::Local(_), .. }) => break,
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
            if scope == CRATE_HIR_ID || !matches!(self.get(scope), Node::Block(_)) {
                return scope;
            }
        }
    }

    pub fn get_foreign_abi(self, hir_id: HirId) -> Abi {
        let parent = self.get_parent_item(hir_id);
        if let Some(node) = self.tcx.hir_owner(parent) {
            if let OwnerNode::Item(Item { kind: ItemKind::ForeignMod { abi, .. }, .. }) = node.node
            {
                return *abi;
            }
        }
        bug!(
            "expected foreign mod or inlined parent, found {}",
            self.node_to_string(HirId::make_owner(parent.def_id))
        )
    }

    pub fn expect_owner(self, def_id: LocalDefId) -> OwnerNode<'hir> {
        self.tcx
            .hir_owner(OwnerId { def_id })
            .unwrap_or_else(|| bug!("expected owner for {:?}", def_id))
            .node
    }

    pub fn expect_item(self, id: LocalDefId) -> &'hir Item<'hir> {
        match self.tcx.hir_owner(OwnerId { def_id: id }) {
            Some(Owner { node: OwnerNode::Item(item), .. }) => item,
            _ => bug!("expected item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_impl_item(self, id: LocalDefId) -> &'hir ImplItem<'hir> {
        match self.tcx.hir_owner(OwnerId { def_id: id }) {
            Some(Owner { node: OwnerNode::ImplItem(item), .. }) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_trait_item(self, id: LocalDefId) -> &'hir TraitItem<'hir> {
        match self.tcx.hir_owner(OwnerId { def_id: id }) {
            Some(Owner { node: OwnerNode::TraitItem(item), .. }) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_variant(self, id: HirId) -> &'hir Variant<'hir> {
        match self.find(id) {
            Some(Node::Variant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(self, id: OwnerId) -> &'hir ForeignItem<'hir> {
        match self.tcx.hir_owner(id) {
            Some(Owner { node: OwnerNode::ForeignItem(item), .. }) => item,
            _ => {
                bug!(
                    "expected foreign item, found {}",
                    self.node_to_string(HirId::make_owner(id.def_id))
                )
            }
        }
    }

    pub fn expect_expr(self, id: HirId) -> &'hir Expr<'hir> {
        match self.find(id) {
            Some(Node::Expr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id)),
        }
    }

    #[inline]
    fn opt_ident(self, id: HirId) -> Option<Ident> {
        match self.get(id) {
            Node::Pat(&Pat { kind: PatKind::Binding(_, _, ident, _), .. }) => Some(ident),
            // A `Ctor` doesn't have an identifier itself, but its parent
            // struct/variant does. Compare with `hir::Map::opt_span`.
            Node::Ctor(..) => match self.find_parent(id)? {
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
        self.opt_span(hir_id)
            .unwrap_or_else(|| bug!("hir::map::Map::span: id not in map: {:?}", hir_id))
    }

    pub fn opt_span(self, hir_id: HirId) -> Option<Span> {
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

        let span = match self.find(hir_id)? {
            // Function-like.
            Node::Item(Item { kind: ItemKind::Fn(sig, ..), span: outer_span, .. })
            | Node::TraitItem(TraitItem {
                kind: TraitItemKind::Fn(sig, ..),
                span: outer_span,
                ..
            })
            | Node::ImplItem(ImplItem {
                kind: ImplItemKind::Fn(sig, ..), span: outer_span, ..
            }) => {
                // Ensure that the returned span has the item's SyntaxContext, and not the
                // SyntaxContext of the visibility.
                sig.span.find_ancestor_in_same_ctxt(*outer_span).unwrap_or(*outer_span)
            }
            // Constants and Statics.
            Node::Item(Item {
                kind:
                    ItemKind::Const(ty, ..)
                    | ItemKind::Static(ty, ..)
                    | ItemKind::Impl(Impl { self_ty: ty, .. }),
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
            Node::ForeignItem(item) => match item.kind {
                ForeignItemKind::Fn(decl, _, _) => until_within(item.span, decl.output.span()),
                _ => named_span(item.span, item.ident, None),
            },
            Node::Ctor(_) => return self.opt_span(self.parent_id(hir_id)),
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
        Some(span)
    }

    /// Like `hir.span()`, but includes the body of items
    /// (instead of just the item header)
    pub fn span_with_body(self, hir_id: HirId) -> Span {
        match self.get(hir_id) {
            Node::Param(param) => param.span,
            Node::Item(item) => item.span,
            Node::ForeignItem(foreign_item) => foreign_item.span,
            Node::TraitItem(trait_item) => trait_item.span,
            Node::ImplItem(impl_item) => impl_item.span,
            Node::Variant(variant) => variant.span,
            Node::Field(field) => field.span,
            Node::AnonConst(constant) => self.body(constant.body).value.span,
            Node::Expr(expr) => expr.span,
            Node::ExprField(field) => field.span,
            Node::Stmt(stmt) => stmt.span,
            Node::PathSegment(seg) => {
                let ident_span = seg.ident.span;
                ident_span
                    .with_hi(seg.args.map_or_else(|| ident_span.hi(), |args| args.span_ext.hi()))
            }
            Node::Ty(ty) => ty.span,
            Node::TypeBinding(tb) => tb.span,
            Node::TraitRef(tr) => tr.path.span,
            Node::Pat(pat) => pat.span,
            Node::PatField(field) => field.span,
            Node::Arm(arm) => arm.span,
            Node::Block(block) => block.span,
            Node::Ctor(..) => self.span_with_body(self.parent_id(hir_id)),
            Node::Lifetime(lifetime) => lifetime.ident.span,
            Node::GenericParam(param) => param.span,
            Node::Infer(i) => i.span,
            Node::Local(local) => local.span,
            Node::Crate(item) => item.spans.inner_span,
        }
    }

    pub fn span_if_local(self, id: DefId) -> Option<Span> {
        if id.is_local() { Some(self.tcx.def_span(id)) } else { None }
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
        match self.get_parent(anon_const) {
            Node::GenericParam(GenericParam {
                def_id: param_id,
                kind: GenericParamKind::Const { .. },
                ..
            }) => Some(*param_id),
            _ => None,
        }
    }
}

impl<'hir> intravisit::Map<'hir> for Map<'hir> {
    fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        (*self).find(hir_id)
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

pub(super) fn crate_hash(tcx: TyCtxt<'_>, crate_num: CrateNum) -> Svh {
    debug_assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir_crate(());
    let hir_body_hash = krate.hir_hash;

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
        .map(|source_file| source_file.name_hash)
        .collect();

    source_file_names.sort_unstable();

    let crate_hash: Fingerprint = tcx.with_stable_hashing_context(|mut hcx| {
        let mut stable_hasher = StableHasher::new();
        hir_body_hash.hash_stable(&mut hcx, &mut stable_hasher);
        upstream_crates.hash_stable(&mut hcx, &mut stable_hasher);
        source_file_names.hash_stable(&mut hcx, &mut stable_hasher);
        if tcx.sess.opts.incremental_relative_spans() {
            let definitions = tcx.definitions_untracked();
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
        tcx.sess.local_stable_crate_id().hash_stable(&mut hcx, &mut stable_hasher);
        // Hash visibility information since it does not appear in HIR.
        resolutions.visibilities.hash_stable(&mut hcx, &mut stable_hasher);
        resolutions.has_pub_restricted.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    });

    Svh::new(crate_hash.to_smaller_hash())
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
    let id_str = format!(" (hir_id={})", id);

    let path_str = |def_id: LocalDefId| map.tcx.def_path_str(def_id.to_def_id());

    let span_str = || map.tcx.sess.source_map().span_to_snippet(map.span(id)).unwrap_or_default();
    let node_str = |prefix| format!("{} {}{}", prefix, span_str(), id_str);

    match map.find(id) {
        Some(Node::Item(item)) => {
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
                ItemKind::OpaqueTy(ref opaque) => {
                    if opaque.in_trait {
                        "opaque type in trait"
                    } else {
                        "opaque type"
                    }
                }
                ItemKind::Enum(..) => "enum",
                ItemKind::Struct(..) => "struct",
                ItemKind::Union(..) => "union",
                ItemKind::Trait(..) => "trait",
                ItemKind::TraitAlias(..) => "trait alias",
                ItemKind::Impl { .. } => "impl",
            };
            format!("{} {}{}", item_str, path_str(item.owner_id.def_id), id_str)
        }
        Some(Node::ForeignItem(item)) => {
            format!("foreign item {}{}", path_str(item.owner_id.def_id), id_str)
        }
        Some(Node::ImplItem(ii)) => {
            let kind = match ii.kind {
                ImplItemKind::Const(..) => "assoc const",
                ImplItemKind::Fn(..) => "method",
                ImplItemKind::Type(_) => "assoc type",
            };
            format!("{} {} in {}{}", kind, ii.ident, path_str(ii.owner_id.def_id), id_str)
        }
        Some(Node::TraitItem(ti)) => {
            let kind = match ti.kind {
                TraitItemKind::Const(..) => "assoc constant",
                TraitItemKind::Fn(..) => "trait method",
                TraitItemKind::Type(..) => "assoc type",
            };

            format!("{} {} in {}{}", kind, ti.ident, path_str(ti.owner_id.def_id), id_str)
        }
        Some(Node::Variant(ref variant)) => {
            format!("variant {} in {}{}", variant.ident, path_str(variant.def_id), id_str)
        }
        Some(Node::Field(ref field)) => {
            format!("field {} in {}{}", field.ident, path_str(field.def_id), id_str)
        }
        Some(Node::AnonConst(_)) => node_str("const"),
        Some(Node::Expr(_)) => node_str("expr"),
        Some(Node::ExprField(_)) => node_str("expr field"),
        Some(Node::Stmt(_)) => node_str("stmt"),
        Some(Node::PathSegment(_)) => node_str("path segment"),
        Some(Node::Ty(_)) => node_str("type"),
        Some(Node::TypeBinding(_)) => node_str("type binding"),
        Some(Node::TraitRef(_)) => node_str("trait ref"),
        Some(Node::Pat(_)) => node_str("pat"),
        Some(Node::PatField(_)) => node_str("pattern field"),
        Some(Node::Param(_)) => node_str("param"),
        Some(Node::Arm(_)) => node_str("arm"),
        Some(Node::Block(_)) => node_str("block"),
        Some(Node::Infer(_)) => node_str("infer"),
        Some(Node::Local(_)) => node_str("local"),
        Some(Node::Ctor(ctor)) => format!(
            "ctor {}{}",
            ctor.ctor_def_id().map_or("<missing path>".into(), |def_id| path_str(def_id)),
            id_str
        ),
        Some(Node::Lifetime(_)) => node_str("lifetime"),
        Some(Node::GenericParam(ref param)) => {
            format!("generic_param {}{}", path_str(param.def_id), id_str)
        }
        Some(Node::Crate(..)) => String::from("root_crate"),
        None => format!("unknown node{}", id_str),
    }
}

pub(super) fn hir_module_items(tcx: TyCtxt<'_>, module_id: LocalDefId) -> ModuleItems {
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
        ..
    } = collector;
    return ModuleItems {
        submodules: submodules.into_boxed_slice(),
        items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
    };
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
        ..
    } = collector;

    return ModuleItems {
        submodules: submodules.into_boxed_slice(),
        items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
    };
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
        }
    }
}

impl<'hir> Visitor<'hir> for ItemCollector<'hir> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'hir Item<'hir>) {
        if associated_body(Node::Item(item)).is_some() {
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

    fn visit_expr(&mut self, ex: &'hir Expr<'hir>) {
        if let ExprKind::Closure(closure) = ex.kind {
            self.body_owners.push(closure.def_id);
        }
        intravisit::walk_expr(self, ex)
    }

    fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
        if associated_body(Node::TraitItem(item)).is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.trait_items.push(item.trait_item_id());
        intravisit::walk_trait_item(self, item)
    }

    fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
        if associated_body(Node::ImplItem(item)).is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.impl_items.push(item.impl_item_id());
        intravisit::walk_impl_item(self, item)
    }
}
