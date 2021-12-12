use crate::hir::{ModuleItems, Owner};
use crate::ty::TyCtxt;
use rustc_ast as ast;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{par_for_each_in, Send, Sync};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::*;
use rustc_index::vec::Idx;
use rustc_span::def_id::StableCrateId;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;
use std::collections::VecDeque;

fn fn_decl<'hir>(node: Node<'hir>) -> Option<&'hir FnDecl<'hir>> {
    match node {
        Node::Item(Item { kind: ItemKind::Fn(sig, _, _), .. })
        | Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(sig, _), .. })
        | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(sig, _), .. }) => Some(&sig.decl),
        Node::Expr(Expr { kind: ExprKind::Closure(_, fn_decl, ..), .. })
        | Node::ForeignItem(ForeignItem { kind: ForeignItemKind::Fn(fn_decl, ..), .. }) => {
            Some(fn_decl)
        }
        _ => None,
    }
}

pub fn fn_sig<'hir>(node: Node<'hir>) -> Option<&'hir FnSig<'hir>> {
    match &node {
        Node::Item(Item { kind: ItemKind::Fn(sig, _, _), .. })
        | Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(sig, _), .. })
        | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(sig, _), .. }) => Some(sig),
        _ => None,
    }
}

pub fn associated_body<'hir>(node: Node<'hir>) -> Option<BodyId> {
    match node {
        Node::Item(Item {
            kind: ItemKind::Const(_, body) | ItemKind::Static(.., body) | ItemKind::Fn(.., body),
            ..
        })
        | Node::TraitItem(TraitItem {
            kind:
                TraitItemKind::Const(_, Some(body)) | TraitItemKind::Fn(_, TraitFn::Provided(body)),
            ..
        })
        | Node::ImplItem(ImplItem {
            kind: ImplItemKind::Const(_, body) | ImplItemKind::Fn(_, body),
            ..
        })
        | Node::Expr(Expr { kind: ExprKind::Closure(.., body, _, _), .. }) => Some(*body),

        Node::AnonConst(constant) => Some(constant.body),

        _ => None,
    }
}

fn is_body_owner<'hir>(node: Node<'hir>, hir_id: HirId) -> bool {
    match associated_body(node) {
        Some(b) => b.hir_id == hir_id,
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
            if let Some(node) = self.map.find(parent_id) {
                return Some((parent_id, node));
            }
            // If this `HirId` doesn't have an entry, skip it and look for its `parent_id`.
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
    type Item = (HirId, OwnerNode<'hir>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id.local_id.index() != 0 {
            self.current_id.local_id = ItemLocalId::new(0);
            if let Some(node) = self.map.tcx.hir_owner(self.current_id.owner) {
                return Some((self.current_id, node.node));
            }
        }
        if self.current_id == CRATE_HIR_ID {
            return None;
        }
        loop {
            // There are nodes that do not have entries, so we need to skip them.
            let parent_id = self.map.def_key(self.current_id.owner).parent;

            let parent_id = parent_id.map_or(CRATE_HIR_ID.owner, |local_def_index| {
                let def_id = LocalDefId { local_def_index };
                self.map.local_def_id_to_hir_id(def_id).owner
            });
            self.current_id = HirId::make_owner(parent_id);

            // If this `HirId` doesn't have an entry, skip it and look for its `parent_id`.
            if let Some(node) = self.map.tcx.hir_owner(self.current_id.owner) {
                return Some((self.current_id, node.node));
            }
        }
    }
}

impl<'hir> Map<'hir> {
    pub fn krate(&self) -> &'hir Crate<'hir> {
        self.tcx.hir_crate(())
    }

    pub fn root_module(&self) -> &'hir Mod<'hir> {
        match self.tcx.hir_owner(CRATE_DEF_ID).map(|o| o.node) {
            Some(OwnerNode::Crate(item)) => item,
            _ => bug!(),
        }
    }

    pub fn items(&self) -> impl Iterator<Item = &'hir Item<'hir>> + 'hir {
        let krate = self.krate();
        krate.owners.iter().filter_map(|owner| match owner.as_ref()?.node() {
            OwnerNode::Item(item) => Some(item),
            _ => None,
        })
    }

    pub fn def_key(&self, def_id: LocalDefId) -> DefKey {
        // Accessing the DefKey is ok, since it is part of DefPathHash.
        self.tcx.untracked_resolutions.definitions.def_key(def_id)
    }

    pub fn def_path_from_hir_id(&self, id: HirId) -> Option<DefPath> {
        self.opt_local_def_id(id).map(|def_id| self.def_path(def_id))
    }

    pub fn def_path(&self, def_id: LocalDefId) -> DefPath {
        // Accessing the DefPath is ok, since it is part of DefPathHash.
        self.tcx.untracked_resolutions.definitions.def_path(def_id)
    }

    #[inline]
    pub fn def_path_hash(self, def_id: LocalDefId) -> DefPathHash {
        // Accessing the DefPathHash is ok, it is incr. comp. stable.
        self.tcx.untracked_resolutions.definitions.def_path_hash(def_id)
    }

    #[inline]
    pub fn local_def_id(&self, hir_id: HirId) -> LocalDefId {
        self.opt_local_def_id(hir_id).unwrap_or_else(|| {
            bug!(
                "local_def_id: no entry for `{:?}`, which has a map of `{:?}`",
                hir_id,
                self.find(hir_id)
            )
        })
    }

    #[inline]
    pub fn opt_local_def_id(&self, hir_id: HirId) -> Option<LocalDefId> {
        // FIXME(#85914) is this access safe for incr. comp.?
        self.tcx.untracked_resolutions.definitions.opt_hir_id_to_local_def_id(hir_id)
    }

    #[inline]
    pub fn local_def_id_to_hir_id(&self, def_id: LocalDefId) -> HirId {
        // FIXME(#85914) is this access safe for incr. comp.?
        self.tcx.untracked_resolutions.definitions.local_def_id_to_hir_id(def_id)
    }

    pub fn iter_local_def_id(&self) -> impl Iterator<Item = LocalDefId> + '_ {
        // Create a dependency to the crate to be sure we reexcute this when the amount of
        // definitions change.
        self.tcx.ensure().hir_crate(());
        self.tcx.untracked_resolutions.definitions.iter_local_def_id()
    }

    pub fn opt_def_kind(&self, local_def_id: LocalDefId) -> Option<DefKind> {
        let hir_id = self.local_def_id_to_hir_id(local_def_id);
        let def_kind = match self.find(hir_id)? {
            Node::Item(item) => match item.kind {
                ItemKind::Static(..) => DefKind::Static,
                ItemKind::Const(..) => DefKind::Const,
                ItemKind::Fn(..) => DefKind::Fn,
                ItemKind::Macro(..) => DefKind::Macro(MacroKind::Bang),
                ItemKind::Mod(..) => DefKind::Mod,
                ItemKind::OpaqueTy(..) => DefKind::OpaqueTy,
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
                ImplItemKind::Fn(..) => DefKind::AssocFn,
                ImplItemKind::TyAlias(..) => DefKind::AssocTy,
            },
            Node::Variant(_) => DefKind::Variant,
            Node::Ctor(variant_data) => {
                // FIXME(eddyb) is this even possible, if we have a `Node::Ctor`?
                assert_ne!(variant_data.ctor_hir_id(), None);

                let ctor_of = match self.find(self.get_parent_node(hir_id)) {
                    Some(Node::Item(..)) => def::CtorOf::Struct,
                    Some(Node::Variant(..)) => def::CtorOf::Variant,
                    _ => unreachable!(),
                };
                DefKind::Ctor(ctor_of, def::CtorKind::from_hir(variant_data))
            }
            Node::AnonConst(_) => {
                let inline = match self.find(self.get_parent_node(hir_id)) {
                    Some(Node::Expr(&Expr {
                        kind: ExprKind::ConstBlock(ref anon_const), ..
                    })) if anon_const.hir_id == hir_id => true,
                    _ => false,
                };
                if inline { DefKind::InlineConst } else { DefKind::AnonConst }
            }
            Node::Field(_) => DefKind::Field,
            Node::Expr(expr) => match expr.kind {
                ExprKind::Closure(.., None) => DefKind::Closure,
                ExprKind::Closure(.., Some(_)) => DefKind::Generator,
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
            | Node::Infer(_)
            | Node::TraitRef(_)
            | Node::Pat(_)
            | Node::Binding(_)
            | Node::Local(_)
            | Node::Param(_)
            | Node::Arm(_)
            | Node::Lifetime(_)
            | Node::Visibility(_)
            | Node::Block(_) => return None,
        };
        Some(def_kind)
    }

    pub fn def_kind(&self, local_def_id: LocalDefId) -> DefKind {
        self.opt_def_kind(local_def_id)
            .unwrap_or_else(|| bug!("def_kind: unsupported node: {:?}", local_def_id))
    }

    pub fn find_parent_node(&self, id: HirId) -> Option<HirId> {
        if id.local_id == ItemLocalId::from_u32(0) {
            Some(self.tcx.hir_owner_parent(id.owner))
        } else {
            let owner = self.tcx.hir_owner_nodes(id.owner)?;
            let node = owner.nodes[id.local_id].as_ref()?;
            let hir_id = HirId { owner: id.owner, local_id: node.parent };
            Some(hir_id)
        }
    }

    pub fn get_parent_node(&self, hir_id: HirId) -> HirId {
        self.find_parent_node(hir_id).unwrap()
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    pub fn find(&self, id: HirId) -> Option<Node<'hir>> {
        if id.local_id == ItemLocalId::from_u32(0) {
            let owner = self.tcx.hir_owner(id.owner)?;
            Some(owner.node.into())
        } else {
            let owner = self.tcx.hir_owner_nodes(id.owner)?;
            let node = owner.nodes[id.local_id].as_ref()?;
            Some(node.node)
        }
    }

    /// Retrieves the `Node` corresponding to `id`, panicking if it cannot be found.
    pub fn get(&self, id: HirId) -> Node<'hir> {
        self.find(id).unwrap_or_else(|| bug!("couldn't find hir id {} in the HIR map", id))
    }

    pub fn get_if_local(&self, id: DefId) -> Option<Node<'hir>> {
        id.as_local().and_then(|id| self.find(self.local_def_id_to_hir_id(id)))
    }

    pub fn get_generics(&self, id: DefId) -> Option<&'hir Generics<'hir>> {
        let id = id.as_local()?;
        let node = self.tcx.hir_owner(id)?;
        match node.node {
            OwnerNode::ImplItem(impl_item) => Some(&impl_item.generics),
            OwnerNode::TraitItem(trait_item) => Some(&trait_item.generics),
            OwnerNode::Item(Item {
                kind:
                    ItemKind::Fn(_, generics, _)
                    | ItemKind::TyAlias(_, generics)
                    | ItemKind::Enum(_, generics)
                    | ItemKind::Struct(_, generics)
                    | ItemKind::Union(_, generics)
                    | ItemKind::Trait(_, _, generics, ..)
                    | ItemKind::TraitAlias(generics, _)
                    | ItemKind::Impl(Impl { generics, .. }),
                ..
            }) => Some(generics),
            _ => None,
        }
    }

    pub fn item(&self, id: ItemId) -> &'hir Item<'hir> {
        self.tcx.hir_owner(id.def_id).unwrap().node.expect_item()
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.tcx.hir_owner(id.def_id).unwrap().node.expect_trait_item()
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.tcx.hir_owner(id.def_id).unwrap().node.expect_impl_item()
    }

    pub fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.tcx.hir_owner(id.def_id).unwrap().node.expect_foreign_item()
    }

    pub fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        self.tcx.hir_owner_nodes(id.hir_id.owner).unwrap().bodies[&id.hir_id.local_id]
    }

    pub fn fn_decl_by_hir_id(&self, hir_id: HirId) -> Option<&'hir FnDecl<'hir>> {
        if let Some(node) = self.find(hir_id) {
            fn_decl(node)
        } else {
            bug!("no node for hir_id `{}`", hir_id)
        }
    }

    pub fn fn_sig_by_hir_id(&self, hir_id: HirId) -> Option<&'hir FnSig<'hir>> {
        if let Some(node) = self.find(hir_id) {
            fn_sig(node)
        } else {
            bug!("no node for hir_id `{}`", hir_id)
        }
    }

    pub fn enclosing_body_owner(&self, hir_id: HirId) -> HirId {
        for (parent, _) in self.parent_iter(hir_id) {
            if let Some(body) = self.maybe_body_owned_by(parent) {
                return self.body_owner(body);
            }
        }

        bug!("no `enclosing_body_owner` for hir_id `{}`", hir_id);
    }

    /// Returns the `HirId` that corresponds to the definition of
    /// which this is the body of, i.e., a `fn`, `const` or `static`
    /// item (possibly associated), a closure, or a `hir::AnonConst`.
    pub fn body_owner(&self, BodyId { hir_id }: BodyId) -> HirId {
        let parent = self.get_parent_node(hir_id);
        assert!(self.find(parent).map_or(false, |n| is_body_owner(n, hir_id)));
        parent
    }

    pub fn body_owner_def_id(&self, id: BodyId) -> LocalDefId {
        self.local_def_id(self.body_owner(id))
    }

    /// Given a `HirId`, returns the `BodyId` associated with it,
    /// if the node is a body owner, otherwise returns `None`.
    pub fn maybe_body_owned_by(&self, hir_id: HirId) -> Option<BodyId> {
        self.find(hir_id).map(associated_body).flatten()
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

    pub fn body_param_names(&self, id: BodyId) -> impl Iterator<Item = Ident> + 'hir {
        self.body(id).params.iter().map(|arg| match arg.pat.kind {
            PatKind::Binding(_, _, ident, _) => ident,
            _ => Ident::empty(),
        })
    }

    /// Returns the `BodyOwnerKind` of this `LocalDefId`.
    ///
    /// Panics if `LocalDefId` does not have an associated body.
    pub fn body_owner_kind(&self, id: HirId) -> BodyOwnerKind {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Const(..), .. })
            | Node::TraitItem(&TraitItem { kind: TraitItemKind::Const(..), .. })
            | Node::ImplItem(&ImplItem { kind: ImplItemKind::Const(..), .. })
            | Node::AnonConst(_) => BodyOwnerKind::Const,
            Node::Ctor(..)
            | Node::Item(&Item { kind: ItemKind::Fn(..), .. })
            | Node::TraitItem(&TraitItem { kind: TraitItemKind::Fn(..), .. })
            | Node::ImplItem(&ImplItem { kind: ImplItemKind::Fn(..), .. }) => BodyOwnerKind::Fn,
            Node::Item(&Item { kind: ItemKind::Static(_, m, _), .. }) => BodyOwnerKind::Static(m),
            Node::Expr(&Expr { kind: ExprKind::Closure(..), .. }) => BodyOwnerKind::Closure,
            node => bug!("{:#?} is not a body node", node),
        }
    }

    /// Returns the `ConstContext` of the body associated with this `LocalDefId`.
    ///
    /// Panics if `LocalDefId` does not have an associated body.
    ///
    /// This should only be used for determining the context of a body, a return
    /// value of `Some` does not always suggest that the owner of the body is `const`,
    /// just that it has to be checked as if it were.
    pub fn body_const_context(&self, did: LocalDefId) -> Option<ConstContext> {
        let hir_id = self.local_def_id_to_hir_id(did);
        let ccx = match self.body_owner_kind(hir_id) {
            BodyOwnerKind::Const => ConstContext::Const,
            BodyOwnerKind::Static(mt) => ConstContext::Static(mt),

            BodyOwnerKind::Fn if self.tcx.is_constructor(did.to_def_id()) => return None,
            BodyOwnerKind::Fn if self.tcx.is_const_fn_raw(did.to_def_id()) => ConstContext::ConstFn,
            BodyOwnerKind::Fn
                if self.tcx.has_attr(did.to_def_id(), sym::default_method_body_is_const) =>
            {
                ConstContext::ConstFn
            }
            BodyOwnerKind::Fn | BodyOwnerKind::Closure => return None,
        };

        Some(ccx)
    }

    /// Returns an iterator of the `DefId`s for all body-owners in this
    /// crate. If you would prefer to iterate over the bodies
    /// themselves, you can do `self.hir().krate().body_ids.iter()`.
    pub fn body_owners(self) -> impl Iterator<Item = LocalDefId> + 'hir {
        self.krate()
            .owners
            .iter_enumerated()
            .flat_map(move |(owner, owner_info)| {
                let bodies = &owner_info.as_ref()?.nodes.bodies;
                Some(bodies.iter().map(move |&(local_id, _)| {
                    let hir_id = HirId { owner, local_id };
                    let body_id = BodyId { hir_id };
                    self.body_owner_def_id(body_id)
                }))
            })
            .flatten()
    }

    pub fn par_body_owners<F: Fn(LocalDefId) + Sync + Send>(self, f: F) {
        use rustc_data_structures::sync::{par_iter, ParallelIterator};
        #[cfg(parallel_compiler)]
        use rustc_rayon::iter::IndexedParallelIterator;

        par_iter(&self.krate().owners.raw).enumerate().for_each(|(owner, owner_info)| {
            let owner = LocalDefId::new(owner);
            if let Some(owner_info) = owner_info {
                par_iter(owner_info.nodes.bodies.range(..)).for_each(|(local_id, _)| {
                    let hir_id = HirId { owner, local_id: *local_id };
                    let body_id = BodyId { hir_id };
                    f(self.body_owner_def_id(body_id))
                })
            }
        });
    }

    pub fn ty_param_owner(&self, id: HirId) -> HirId {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Trait(..) | ItemKind::TraitAlias(..), .. }) => id,
            Node::GenericParam(_) => self.get_parent_node(id),
            _ => bug!("ty_param_owner: {} not a type parameter", self.node_to_string(id)),
        }
    }

    pub fn ty_param_name(&self, id: HirId) -> Symbol {
        match self.get(id) {
            Node::Item(&Item { kind: ItemKind::Trait(..) | ItemKind::TraitAlias(..), .. }) => {
                kw::SelfUpper
            }
            Node::GenericParam(param) => param.name.ident().name,
            _ => bug!("ty_param_name: {} not a type parameter", self.node_to_string(id)),
        }
    }

    pub fn trait_impls(&self, trait_did: DefId) -> &'hir [LocalDefId] {
        self.tcx.all_local_trait_impls(()).get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    /// Gets the attributes on the crate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'hir [ast::Attribute] {
        self.attrs(CRATE_HIR_ID)
    }

    pub fn get_module(&self, module: LocalDefId) -> (&'hir Mod<'hir>, Span, HirId) {
        let hir_id = HirId::make_owner(module);
        match self.tcx.hir_owner(module).map(|o| o.node) {
            Some(OwnerNode::Item(&Item { span, kind: ItemKind::Mod(ref m), .. })) => {
                (m, span, hir_id)
            }
            Some(OwnerNode::Crate(item)) => (item, item.inner, hir_id),
            node => panic!("not a module: {:?}", node),
        }
    }

    /// Walks the contents of a crate. See also `Crate::visit_all_items`.
    pub fn walk_toplevel_module(self, visitor: &mut impl Visitor<'hir>) {
        let (top_mod, span, hir_id) = self.get_module(CRATE_DEF_ID);
        visitor.visit_mod(top_mod, span, hir_id);
    }

    /// Walks the attributes in a crate.
    pub fn walk_attributes(self, visitor: &mut impl Visitor<'hir>) {
        let krate = self.krate();
        for (owner, info) in krate.owners.iter_enumerated() {
            if let Some(info) = info {
                for (local_id, attrs) in info.attrs.map.iter() {
                    let id = HirId { owner, local_id: *local_id };
                    for a in *attrs {
                        visitor.visit_attribute(id, a)
                    }
                }
            }
        }
    }

    /// Visits all items in the crate in some deterministic (but
    /// unspecified) order. If you just need to process every item,
    /// but don't care about nesting, this method is the best choice.
    ///
    /// If you do care about nesting -- usually because your algorithm
    /// follows lexical scoping rules -- then you want a different
    /// approach. You should override `visit_nested_item` in your
    /// visitor and then call `intravisit::walk_crate` instead.
    pub fn visit_all_item_likes<V>(&self, visitor: &mut V)
    where
        V: itemlikevisit::ItemLikeVisitor<'hir>,
    {
        let krate = self.krate();
        for owner in krate.owners.iter().filter_map(Option::as_ref) {
            match owner.node() {
                OwnerNode::Item(item) => visitor.visit_item(item),
                OwnerNode::ForeignItem(item) => visitor.visit_foreign_item(item),
                OwnerNode::ImplItem(item) => visitor.visit_impl_item(item),
                OwnerNode::TraitItem(item) => visitor.visit_trait_item(item),
                OwnerNode::Crate(_) => {}
            }
        }
    }

    /// A parallel version of `visit_all_item_likes`.
    pub fn par_visit_all_item_likes<V>(&self, visitor: &V)
    where
        V: itemlikevisit::ParItemLikeVisitor<'hir> + Sync + Send,
    {
        let krate = self.krate();
        par_for_each_in(&krate.owners.raw, |owner| match owner.as_ref().map(OwnerInfo::node) {
            Some(OwnerNode::Item(item)) => visitor.visit_item(item),
            Some(OwnerNode::ForeignItem(item)) => visitor.visit_foreign_item(item),
            Some(OwnerNode::ImplItem(item)) => visitor.visit_impl_item(item),
            Some(OwnerNode::TraitItem(item)) => visitor.visit_trait_item(item),
            Some(OwnerNode::Crate(_)) | None => {}
        })
    }

    pub fn visit_item_likes_in_module<V>(&self, module: LocalDefId, visitor: &mut V)
    where
        V: ItemLikeVisitor<'hir>,
    {
        let module = self.tcx.hir_module_items(module);

        for id in module.items.iter() {
            visitor.visit_item(self.item(*id));
        }

        for id in module.trait_items.iter() {
            visitor.visit_trait_item(self.trait_item(*id));
        }

        for id in module.impl_items.iter() {
            visitor.visit_impl_item(self.impl_item(*id));
        }

        for id in module.foreign_items.iter() {
            visitor.visit_foreign_item(self.foreign_item(*id));
        }
    }

    pub fn for_each_module(&self, f: impl Fn(LocalDefId)) {
        let mut queue = VecDeque::new();
        queue.push_back(CRATE_DEF_ID);

        while let Some(id) = queue.pop_front() {
            f(id);
            let items = self.tcx.hir_module_items(id);
            queue.extend(items.submodules.iter().copied())
        }
    }

    #[cfg(not(parallel_compiler))]
    #[inline]
    pub fn par_for_each_module(&self, f: impl Fn(LocalDefId)) {
        self.for_each_module(f)
    }

    #[cfg(parallel_compiler)]
    pub fn par_for_each_module(&self, f: impl Fn(LocalDefId) + Sync) {
        use rustc_data_structures::sync::{par_iter, ParallelIterator};
        par_iter_submodules(self.tcx, CRATE_DEF_ID, &f);

        fn par_iter_submodules<F>(tcx: TyCtxt<'_>, module: LocalDefId, f: &F)
        where
            F: Fn(LocalDefId) + Sync,
        {
            (*f)(module);
            let items = tcx.hir_module_items(module);
            par_iter(&items.submodules[..]).for_each(|&sm| par_iter_submodules(tcx, sm, f));
        }
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `get_parent_node`.
    pub fn parent_iter(self, current_id: HirId) -> ParentHirIterator<'hir> {
        ParentHirIterator { current_id, map: self }
    }

    /// Returns an iterator for the nodes in the ancestor tree of the `current_id`
    /// until the crate root is reached. Prefer this over your own loop using `get_parent_node`.
    pub fn parent_owner_iter(self, current_id: HirId) -> ParentOwnerIterator<'hir> {
        ParentOwnerIterator { current_id, map: self }
    }

    /// Checks if the node is left-hand side of an assignment.
    pub fn is_lhs(&self, id: HirId) -> bool {
        match self.find(self.get_parent_node(id)) {
            Some(Node::Expr(expr)) => match expr.kind {
                ExprKind::Assign(lhs, _rhs, _span) => lhs.hir_id == id,
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether the expression pointed at by `hir_id` belongs to a `const` evaluation context.
    /// Used exclusively for diagnostics, to avoid suggestion function calls.
    pub fn is_inside_const_context(&self, hir_id: HirId) -> bool {
        self.body_const_context(self.local_def_id(self.enclosing_body_owner(hir_id))).is_some()
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
        if let Some(node) = self.find(id) {
            if let Node::Expr(Expr { kind: ExprKind::Ret(_), .. }) = node {
                // When dealing with `return` statements, we don't care about climbing only tail
                // expressions.
                ignore_tail = true;
            }
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
                | Node::Expr(Expr { kind: ExprKind::Closure(..), .. })
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

    /// Retrieves the `HirId` for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the HIR which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent_item(&self, hir_id: HirId) -> HirId {
        if let Some((hir_id, _node)) = self.parent_owner_iter(hir_id).next() {
            hir_id
        } else {
            CRATE_HIR_ID
        }
    }

    /// Returns the `HirId` of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub(super) fn get_module_parent_node(&self, hir_id: HirId) -> HirId {
        for (hir_id, node) in self.parent_owner_iter(hir_id) {
            if let OwnerNode::Item(&Item { kind: ItemKind::Mod(_), .. }) = node {
                return hir_id;
            }
        }
        CRATE_HIR_ID
    }

    /// When on an if expression, a match arm tail expression or a match arm, give back
    /// the enclosing `if` or `match` expression.
    ///
    /// Used by error reporting when there's a type error in an if or match arm caused by the
    /// expression needing to be unit.
    pub fn get_if_cause(&self, hir_id: HirId) -> Option<&'hir Expr<'hir>> {
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
    pub fn get_enclosing_scope(&self, hir_id: HirId) -> Option<HirId> {
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
    pub fn get_defining_scope(&self, id: HirId) -> HirId {
        let mut scope = id;
        loop {
            scope = self.get_enclosing_scope(scope).unwrap_or(CRATE_HIR_ID);
            if scope == CRATE_HIR_ID || !matches!(self.get(scope), Node::Block(_)) {
                return scope;
            }
        }
    }

    pub fn get_parent_did(&self, id: HirId) -> LocalDefId {
        self.local_def_id(self.get_parent_item(id))
    }

    pub fn get_foreign_abi(&self, hir_id: HirId) -> Abi {
        let parent = self.get_parent_item(hir_id);
        if let Some(node) = self.tcx.hir_owner(self.local_def_id(parent)) {
            if let OwnerNode::Item(Item { kind: ItemKind::ForeignMod { abi, .. }, .. }) = node.node
            {
                return *abi;
            }
        }
        bug!("expected foreign mod or inlined parent, found {}", self.node_to_string(parent))
    }

    pub fn expect_item(&self, id: LocalDefId) -> &'hir Item<'hir> {
        match self.tcx.hir_owner(id) {
            Some(Owner { node: OwnerNode::Item(item), .. }) => item,
            _ => bug!("expected item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_impl_item(&self, id: LocalDefId) -> &'hir ImplItem<'hir> {
        match self.tcx.hir_owner(id) {
            Some(Owner { node: OwnerNode::ImplItem(item), .. }) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_trait_item(&self, id: LocalDefId) -> &'hir TraitItem<'hir> {
        match self.tcx.hir_owner(id) {
            Some(Owner { node: OwnerNode::TraitItem(item), .. }) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(HirId::make_owner(id))),
        }
    }

    pub fn expect_variant(&self, id: HirId) -> &'hir Variant<'hir> {
        match self.find(id) {
            Some(Node::Variant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(&self, id: LocalDefId) -> &'hir ForeignItem<'hir> {
        match self.tcx.hir_owner(id) {
            Some(Owner { node: OwnerNode::ForeignItem(item), .. }) => item,
            _ => {
                bug!("expected foreign item, found {}", self.node_to_string(HirId::make_owner(id)))
            }
        }
    }

    pub fn expect_expr(&self, id: HirId) -> &'hir Expr<'hir> {
        match self.find(id) {
            Some(Node::Expr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id)),
        }
    }

    pub fn opt_name(&self, id: HirId) -> Option<Symbol> {
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

    pub fn name(&self, id: HirId) -> Symbol {
        match self.opt_name(id) {
            Some(name) => name,
            None => bug!("no name for {}", self.node_to_string(id)),
        }
    }

    /// Given a node ID, gets a list of attributes associated with the AST
    /// corresponding to the node-ID.
    pub fn attrs(&self, id: HirId) -> &'hir [ast::Attribute] {
        self.tcx.hir_attrs(id.owner).get(id.local_id)
    }

    /// Gets the span of the definition of the specified HIR node.
    /// This is used by `tcx.get_span`
    pub fn span(&self, hir_id: HirId) -> Span {
        self.opt_span(hir_id)
            .unwrap_or_else(|| bug!("hir::map::Map::span: id not in map: {:?}", hir_id))
    }

    pub fn opt_span(&self, hir_id: HirId) -> Option<Span> {
        let span = match self.find(hir_id)? {
            Node::Param(param) => param.span,
            Node::Item(item) => match &item.kind {
                ItemKind::Fn(sig, _, _) => sig.span,
                _ => item.span,
            },
            Node::ForeignItem(foreign_item) => foreign_item.span,
            Node::TraitItem(trait_item) => match &trait_item.kind {
                TraitItemKind::Fn(sig, _) => sig.span,
                _ => trait_item.span,
            },
            Node::ImplItem(impl_item) => match &impl_item.kind {
                ImplItemKind::Fn(sig, _) => sig.span,
                _ => impl_item.span,
            },
            Node::Variant(variant) => variant.span,
            Node::Field(field) => field.span,
            Node::AnonConst(constant) => self.body(constant.body).value.span,
            Node::Expr(expr) => expr.span,
            Node::Stmt(stmt) => stmt.span,
            Node::PathSegment(seg) => seg.ident.span,
            Node::Ty(ty) => ty.span,
            Node::TraitRef(tr) => tr.path.span,
            Node::Binding(pat) => pat.span,
            Node::Pat(pat) => pat.span,
            Node::Arm(arm) => arm.span,
            Node::Block(block) => block.span,
            Node::Ctor(..) => match self.find(self.get_parent_node(hir_id))? {
                Node::Item(item) => item.span,
                Node::Variant(variant) => variant.span,
                _ => unreachable!(),
            },
            Node::Lifetime(lifetime) => lifetime.span,
            Node::GenericParam(param) => param.span,
            Node::Visibility(&Spanned {
                node: VisibilityKind::Restricted { ref path, .. },
                ..
            }) => path.span,
            Node::Infer(i) => i.span,
            Node::Visibility(v) => bug!("unexpected Visibility {:?}", v),
            Node::Local(local) => local.span,
            Node::Crate(item) => item.inner,
        };
        Some(span)
    }

    /// Like `hir.span()`, but includes the body of function items
    /// (instead of just the function header)
    pub fn span_with_body(&self, hir_id: HirId) -> Span {
        match self.find(hir_id) {
            Some(Node::TraitItem(item)) => item.span,
            Some(Node::ImplItem(impl_item)) => impl_item.span,
            Some(Node::Item(item)) => item.span,
            Some(_) => self.span(hir_id),
            _ => bug!("hir::map::Map::span_with_body: id not in map: {:?}", hir_id),
        }
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        id.as_local().and_then(|id| self.opt_span(self.local_def_id_to_hir_id(id)))
    }

    pub fn res_span(&self, res: Res) -> Option<Span> {
        match res {
            Res::Err => None,
            Res::Local(id) => Some(self.span(id)),
            res => self.span_if_local(res.opt_def_id()?),
        }
    }

    /// Get a representation of this `id` for debugging purposes.
    /// NOTE: Do NOT use this in diagnostics!
    pub fn node_to_string(&self, id: HirId) -> String {
        hir_id_to_string(self, id)
    }

    /// Returns the HirId of `N` in `struct Foo<const N: usize = { ... }>` when
    /// called with the HirId for the `{ ... }` anon const
    pub fn opt_const_param_default_param_hir_id(&self, anon_const: HirId) -> Option<HirId> {
        match self.get(self.get_parent_node(anon_const)) {
            Node::GenericParam(GenericParam {
                hir_id: param_id,
                kind: GenericParamKind::Const { .. },
                ..
            }) => Some(*param_id),
            _ => None,
        }
    }
}

impl<'hir> intravisit::Map<'hir> for Map<'hir> {
    fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        self.find(hir_id)
    }

    fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        self.body(id)
    }

    fn item(&self, id: ItemId) -> &'hir Item<'hir> {
        self.item(id)
    }

    fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        self.trait_item(id)
    }

    fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        self.impl_item(id)
    }

    fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.foreign_item(id)
    }
}

pub(super) fn crate_hash(tcx: TyCtxt<'_>, crate_num: CrateNum) -> Svh {
    debug_assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir_crate(());
    let hir_body_hash = krate.hir_hash;

    let upstream_crates = upstream_crates(tcx);

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

    let mut hcx = tcx.create_stable_hashing_context();
    let mut stable_hasher = StableHasher::new();
    hir_body_hash.hash_stable(&mut hcx, &mut stable_hasher);
    upstream_crates.hash_stable(&mut hcx, &mut stable_hasher);
    source_file_names.hash_stable(&mut hcx, &mut stable_hasher);
    if tcx.sess.opts.debugging_opts.incremental_relative_spans {
        let definitions = &tcx.untracked_resolutions.definitions;
        let mut owner_spans: Vec<_> = krate
            .owners
            .iter_enumerated()
            .filter_map(|(def_id, info)| {
                let _ = info.as_ref()?;
                let def_path_hash = definitions.def_path_hash(def_id);
                let span = definitions.def_span(def_id);
                debug_assert_eq!(span.parent(), None);
                Some((def_path_hash, span))
            })
            .collect();
        owner_spans.sort_unstable_by_key(|bn| bn.0);
        owner_spans.hash_stable(&mut hcx, &mut stable_hasher);
    }
    tcx.sess.opts.dep_tracking_hash(true).hash_stable(&mut hcx, &mut stable_hasher);
    tcx.sess.local_stable_crate_id().hash_stable(&mut hcx, &mut stable_hasher);

    let crate_hash: Fingerprint = stable_hasher.finish();
    Svh::new(crate_hash.to_smaller_hash())
}

fn upstream_crates(tcx: TyCtxt<'_>) -> Vec<(StableCrateId, Svh)> {
    let mut upstream_crates: Vec<_> = tcx
        .crates(())
        .iter()
        .map(|&cnum| {
            let stable_crate_id = tcx.resolutions(()).cstore.stable_crate_id(cnum);
            let hash = tcx.crate_hash(cnum);
            (stable_crate_id, hash)
        })
        .collect();
    upstream_crates.sort_unstable_by_key(|&(stable_crate_id, _)| stable_crate_id);
    upstream_crates
}

fn hir_id_to_string(map: &Map<'_>, id: HirId) -> String {
    let id_str = format!(" (hir_id={})", id);

    let path_str = || {
        // This functionality is used for debugging, try to use `TyCtxt` to get
        // the user-friendly path, otherwise fall back to stringifying `DefPath`.
        crate::ty::tls::with_opt(|tcx| {
            if let Some(tcx) = tcx {
                let def_id = map.local_def_id(id);
                tcx.def_path_str(def_id.to_def_id())
            } else if let Some(path) = map.def_path_from_hir_id(id) {
                path.data.into_iter().map(|elem| elem.to_string()).collect::<Vec<_>>().join("::")
            } else {
                String::from("<missing path>")
            }
        })
    };

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
            ImplItemKind::Fn(..) => format!("method {} in {}{}", ii.ident, path_str(), id_str),
            ImplItemKind::TyAlias(_) => {
                format!("assoc type {} in {}{}", ii.ident, path_str(), id_str)
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
        Some(Node::AnonConst(_)) => node_str("const"),
        Some(Node::Expr(_)) => node_str("expr"),
        Some(Node::Stmt(_)) => node_str("stmt"),
        Some(Node::PathSegment(_)) => node_str("path segment"),
        Some(Node::Ty(_)) => node_str("type"),
        Some(Node::TraitRef(_)) => node_str("trait ref"),
        Some(Node::Binding(_)) => node_str("local"),
        Some(Node::Pat(_)) => node_str("pat"),
        Some(Node::Param(_)) => node_str("param"),
        Some(Node::Arm(_)) => node_str("arm"),
        Some(Node::Block(_)) => node_str("block"),
        Some(Node::Infer(_)) => node_str("infer"),
        Some(Node::Local(_)) => node_str("local"),
        Some(Node::Ctor(..)) => format!("ctor {}{}", path_str(), id_str),
        Some(Node::Lifetime(_)) => node_str("lifetime"),
        Some(Node::GenericParam(ref param)) => format!("generic_param {:?}{}", param, id_str),
        Some(Node::Visibility(ref vis)) => format!("visibility {:?}{}", vis, id_str),
        Some(Node::Crate(..)) => String::from("root_crate"),
        None => format!("unknown node{}", id_str),
    }
}

pub(super) fn hir_module_items(tcx: TyCtxt<'_>, module_id: LocalDefId) -> ModuleItems {
    let mut collector = ModuleCollector {
        tcx,
        submodules: Vec::default(),
        items: Vec::default(),
        trait_items: Vec::default(),
        impl_items: Vec::default(),
        foreign_items: Vec::default(),
    };

    let (hir_mod, span, hir_id) = tcx.hir().get_module(module_id);
    collector.visit_mod(hir_mod, span, hir_id);

    let ModuleCollector { submodules, items, trait_items, impl_items, foreign_items, .. } =
        collector;
    return ModuleItems {
        submodules: submodules.into_boxed_slice(),
        items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
    };

    struct ModuleCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        submodules: Vec<LocalDefId>,
        items: Vec<ItemId>,
        trait_items: Vec<TraitItemId>,
        impl_items: Vec<ImplItemId>,
        foreign_items: Vec<ForeignItemId>,
    }

    impl<'hir> Visitor<'hir> for ModuleCollector<'hir> {
        type Map = Map<'hir>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::All(self.tcx.hir())
        }

        fn visit_item(&mut self, item: &'hir Item<'hir>) {
            self.items.push(item.item_id());
            if let ItemKind::Mod(..) = item.kind {
                // If this declares another module, do not recurse inside it.
                self.submodules.push(item.def_id);
            } else {
                intravisit::walk_item(self, item)
            }
        }

        fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
            self.trait_items.push(item.trait_item_id());
            intravisit::walk_trait_item(self, item)
        }

        fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
            self.impl_items.push(item.impl_item_id());
            intravisit::walk_impl_item(self, item)
        }

        fn visit_foreign_item(&mut self, item: &'hir ForeignItem<'hir>) {
            self.foreign_items.push(item.foreign_item_id());
            intravisit::walk_foreign_item(self, item)
        }
    }
}
