use self::collector::NodeCollector;

use crate::hir::{Owner, OwnerNodes};
use crate::ty::query::Providers;
use crate::ty::TyCtxt;
use rustc_ast as ast;
use rustc_data_structures::svh::Svh;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, Definitions};
use rustc_hir::intravisit;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::*;
use rustc_index::vec::IndexVec;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

pub mod blocks;
mod collector;

/// Represents an entry and its parent `HirId`.
#[derive(Copy, Clone, Debug)]
pub struct Entry<'hir> {
    parent: HirId,
    node: Node<'hir>,
}

impl<'hir> Entry<'hir> {
    fn parent_node(self) -> Option<HirId> {
        match self.node {
            Node::Crate(_) | Node::MacroDef(_) => None,
            _ => Some(self.parent),
        }
    }
}

fn fn_decl<'hir>(node: Node<'hir>) -> Option<&'hir FnDecl<'hir>> {
    match node {
        Node::Item(Item { kind: ItemKind::Fn(sig, _, _), .. })
        | Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(sig, _), .. })
        | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(sig, _), .. }) => Some(&sig.decl),
        Node::Expr(Expr { kind: ExprKind::Closure(_, fn_decl, ..), .. }) => Some(fn_decl),
        _ => None,
    }
}

fn fn_sig<'hir>(node: Node<'hir>) -> Option<&'hir FnSig<'hir>> {
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

pub(super) struct HirOwnerData<'hir> {
    pub(super) signature: Option<&'hir Owner<'hir>>,
    pub(super) with_bodies: Option<&'hir mut OwnerNodes<'hir>>,
}

pub struct IndexedHir<'hir> {
    /// The SVH of the local crate.
    pub crate_hash: Svh,

    pub(super) map: IndexVec<LocalDefId, HirOwnerData<'hir>>,
}

#[derive(Copy, Clone)]
pub struct Map<'hir> {
    pub(super) tcx: TyCtxt<'hir>,
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
    pub fn krate(&self) -> &'hir Crate<'hir> {
        self.tcx.hir_crate(LOCAL_CRATE)
    }

    #[inline]
    pub fn definitions(&self) -> &'hir Definitions {
        &self.tcx.definitions
    }

    pub fn def_key(&self, def_id: LocalDefId) -> DefKey {
        self.tcx.definitions.def_key(def_id)
    }

    pub fn def_path_from_hir_id(&self, id: HirId) -> Option<DefPath> {
        self.opt_local_def_id(id).map(|def_id| self.def_path(def_id))
    }

    pub fn def_path(&self, def_id: LocalDefId) -> DefPath {
        self.tcx.definitions.def_path(def_id)
    }

    #[inline]
    pub fn local_def_id(&self, hir_id: HirId) -> LocalDefId {
        self.opt_local_def_id(hir_id).unwrap_or_else(|| {
            bug!(
                "local_def_id: no entry for `{:?}`, which has a map of `{:?}`",
                hir_id,
                self.find_entry(hir_id)
            )
        })
    }

    #[inline]
    pub fn opt_local_def_id(&self, hir_id: HirId) -> Option<LocalDefId> {
        self.tcx.definitions.opt_hir_id_to_local_def_id(hir_id)
    }

    #[inline]
    pub fn local_def_id_to_hir_id(&self, def_id: LocalDefId) -> HirId {
        self.tcx.definitions.local_def_id_to_hir_id(def_id)
    }

    #[inline]
    pub fn opt_local_def_id_to_hir_id(&self, def_id: LocalDefId) -> Option<HirId> {
        self.tcx.definitions.opt_local_def_id_to_hir_id(def_id)
    }

    pub fn def_kind(&self, local_def_id: LocalDefId) -> DefKind {
        // FIXME(eddyb) support `find` on the crate root.
        if local_def_id.to_def_id().index == CRATE_DEF_INDEX {
            return DefKind::Mod;
        }

        let hir_id = self.local_def_id_to_hir_id(local_def_id);
        match self.get(hir_id) {
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
            Node::AnonConst(_) => DefKind::AnonConst,
            Node::Field(_) => DefKind::Field,
            Node::Expr(expr) => match expr.kind {
                ExprKind::Closure(.., None) => DefKind::Closure,
                ExprKind::Closure(.., Some(_)) => DefKind::Generator,
                _ => bug!("def_kind: unsupported node: {}", self.node_to_string(hir_id)),
            },
            Node::MacroDef(_) => DefKind::Macro(MacroKind::Bang),
            Node::GenericParam(param) => match param.kind {
                GenericParamKind::Lifetime { .. } => DefKind::LifetimeParam,
                GenericParamKind::Type { .. } => DefKind::TyParam,
                GenericParamKind::Const { .. } => DefKind::ConstParam,
            },
            Node::Stmt(_)
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
            | Node::Crate(_) => bug!("def_kind: unsupported node: {}", self.node_to_string(hir_id)),
        }
    }

    fn find_entry(&self, id: HirId) -> Option<Entry<'hir>> {
        if id.local_id == ItemLocalId::from_u32(0) {
            let owner = self.tcx.hir_owner(id.owner);
            owner.map(|owner| Entry { parent: owner.parent, node: owner.node })
        } else {
            let owner = self.tcx.hir_owner_nodes(id.owner);
            owner.and_then(|owner| {
                let node = owner.nodes[id.local_id].as_ref();
                // FIXME(eddyb) use a single generic type insted of having both
                // `Entry` and `ParentedNode`, which are effectively the same.
                // Alternatively, rewrite code using `Entry` to use `ParentedNode`.
                node.map(|node| Entry {
                    parent: HirId { owner: id.owner, local_id: node.parent },
                    node: node.node,
                })
            })
        }
    }

    fn get_entry(&self, id: HirId) -> Entry<'hir> {
        self.find_entry(id).unwrap()
    }

    pub fn item(&self, id: HirId) -> &'hir Item<'hir> {
        match self.find(id).unwrap() {
            Node::Item(item) => item,
            _ => bug!(),
        }
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem<'hir> {
        match self.find(id.hir_id).unwrap() {
            Node::TraitItem(item) => item,
            _ => bug!(),
        }
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem<'hir> {
        match self.find(id.hir_id).unwrap() {
            Node::ImplItem(item) => item,
            _ => bug!(),
        }
    }

    pub fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        match self.find(id.hir_id).unwrap() {
            Node::ForeignItem(item) => item,
            _ => bug!(),
        }
    }

    pub fn body(&self, id: BodyId) -> &'hir Body<'hir> {
        self.tcx.hir_owner_nodes(id.hir_id.owner).unwrap().bodies.get(&id.hir_id.local_id).unwrap()
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
            _ => Ident::new(kw::Invalid, rustc_span::DUMMY_SP),
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
    pub fn body_const_context(&self, did: LocalDefId) -> Option<ConstContext> {
        let hir_id = self.local_def_id_to_hir_id(did);
        let ccx = match self.body_owner_kind(hir_id) {
            BodyOwnerKind::Const => ConstContext::Const,
            BodyOwnerKind::Static(mt) => ConstContext::Static(mt),

            BodyOwnerKind::Fn if self.tcx.is_constructor(did.to_def_id()) => return None,
            BodyOwnerKind::Fn if self.tcx.is_const_fn_raw(did.to_def_id()) => ConstContext::ConstFn,
            BodyOwnerKind::Fn | BodyOwnerKind::Closure => return None,
        };

        Some(ccx)
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

    pub fn trait_impls(&self, trait_did: DefId) -> &'hir [HirId] {
        self.tcx.all_local_trait_impls(LOCAL_CRATE).get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    /// Gets the attributes on the crate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'hir [ast::Attribute] {
        match self.get_entry(CRATE_HIR_ID).node {
            Node::Crate(item) => item.attrs,
            _ => bug!(),
        }
    }

    pub fn get_module(&self, module: LocalDefId) -> (&'hir Mod<'hir>, Span, HirId) {
        let hir_id = self.local_def_id_to_hir_id(module);
        match self.get_entry(hir_id).node {
            Node::Item(&Item { span, kind: ItemKind::Mod(ref m), .. }) => (m, span, hir_id),
            Node::Crate(item) => (&item.module, item.span, hir_id),
            node => panic!("not a module: {:?}", node),
        }
    }

    pub fn visit_item_likes_in_module<V>(&self, module: LocalDefId, visitor: &mut V)
    where
        V: ItemLikeVisitor<'hir>,
    {
        let module = self.tcx.hir_module_items(module);

        for id in &module.items {
            visitor.visit_item(self.expect_item(*id));
        }

        for id in &module.trait_items {
            visitor.visit_trait_item(self.expect_trait_item(id.hir_id));
        }

        for id in &module.impl_items {
            visitor.visit_impl_item(self.expect_impl_item(id.hir_id));
        }

        for id in &module.foreign_items {
            visitor.visit_foreign_item(self.expect_foreign_item(id.hir_id));
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
        self.get_if_local(id).and_then(|node| match &node {
            Node::ImplItem(impl_item) => Some(&impl_item.generics),
            Node::TraitItem(trait_item) => Some(&trait_item.generics),
            Node::Item(Item {
                kind:
                    ItemKind::Fn(_, generics, _)
                    | ItemKind::TyAlias(_, generics)
                    | ItemKind::Enum(_, generics)
                    | ItemKind::Struct(_, generics)
                    | ItemKind::Union(_, generics)
                    | ItemKind::Trait(_, _, generics, ..)
                    | ItemKind::TraitAlias(generics, _)
                    | ItemKind::Impl { generics, .. },
                ..
            }) => Some(generics),
            _ => None,
        })
    }

    /// Retrieves the `Node` corresponding to `id`, returning `None` if cannot be found.
    pub fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        self.find_entry(hir_id).and_then(|entry| {
            if let Node::Crate(..) = entry.node { None } else { Some(entry.node) }
        })
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
        self.get_entry(hir_id).parent_node().unwrap_or(hir_id)
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
        matches!(
            self.find(self.get_parent_node(id)),
            Some(
                Node::Item(_)
                | Node::TraitItem(_)
                | Node::ImplItem(_)
                | Node::Expr(Expr { kind: ExprKind::Closure(..), .. }),
            )
        )
    }

    /// Whether the expression pointed at by `hir_id` belongs to a `const` evaluation context.
    /// Used exclusively for diagnostics, to avoid suggestion function calls.
    pub fn is_inside_const_context(&self, hir_id: HirId) -> bool {
        self.body_const_context(self.local_def_id(self.enclosing_body_owner(hir_id))).is_some()
    }

    /// Whether `hir_id` corresponds to a `mod` or a crate.
    pub fn is_hir_id_module(&self, hir_id: HirId) -> bool {
        matches!(
            self.get_entry(hir_id).node,
            Node::Item(Item { kind: ItemKind::Mod(_), .. }) | Node::Crate(..)
        )
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
        for (hir_id, node) in self.parent_iter(hir_id) {
            match node {
                Node::Crate(_)
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
                Node::Item(_)
                | Node::ForeignItem(_)
                | Node::TraitItem(_)
                | Node::ImplItem(_)
                | Node::Stmt(Stmt { kind: StmtKind::Local(_), .. }) => break,
                Node::Expr(expr @ Expr { kind: ExprKind::Match(..), .. }) => return Some(expr),
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
            if scope == CRATE_HIR_ID {
                return CRATE_HIR_ID;
            }
            match self.get(scope) {
                Node::Block(_) => {}
                _ => break,
            }
        }
        scope
    }

    pub fn get_parent_did(&self, id: HirId) -> LocalDefId {
        self.local_def_id(self.get_parent_item(id))
    }

    pub fn get_foreign_abi(&self, hir_id: HirId) -> Abi {
        let parent = self.get_parent_item(hir_id);
        if let Some(entry) = self.find_entry(parent) {
            if let Entry {
                node: Node::Item(Item { kind: ItemKind::ForeignMod { abi, .. }, .. }),
                ..
            } = entry
            {
                return *abi;
            }
        }
        bug!("expected foreign mod or inlined parent, found {}", self.node_to_string(parent))
    }

    pub fn expect_item(&self, id: HirId) -> &'hir Item<'hir> {
        match self.find(id) {
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
            Some(
                Node::Ctor(vd)
                | Node::Item(Item { kind: ItemKind::Struct(vd, _) | ItemKind::Union(vd, _), .. }),
            ) => vd,
            Some(Node::Variant(variant)) => &variant.data,
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
        let attrs = self.find_entry(id).map(|entry| match entry.node {
            Node::Param(a) => &a.attrs[..],
            Node::Local(l) => &l.attrs[..],
            Node::Item(i) => &i.attrs[..],
            Node::ForeignItem(fi) => &fi.attrs[..],
            Node::TraitItem(ref ti) => &ti.attrs[..],
            Node::ImplItem(ref ii) => &ii.attrs[..],
            Node::Variant(ref v) => &v.attrs[..],
            Node::Field(ref f) => &f.attrs[..],
            Node::Expr(ref e) => &*e.attrs,
            Node::Stmt(ref s) => s.kind.attrs(|id| self.item(id.id)),
            Node::Arm(ref a) => &*a.attrs,
            Node::GenericParam(param) => &param.attrs[..],
            // Unit/tuple structs/variants take the attributes straight from
            // the struct/variant definition.
            Node::Ctor(..) => self.attrs(self.get_parent_item(id)),
            Node::Crate(item) => &item.attrs[..],
            Node::MacroDef(def) => def.attrs,
            Node::AnonConst(..)
            | Node::PathSegment(..)
            | Node::Ty(..)
            | Node::Pat(..)
            | Node::Binding(..)
            | Node::TraitRef(..)
            | Node::Block(..)
            | Node::Lifetime(..)
            | Node::Visibility(..) => &[],
        });
        attrs.unwrap_or(&[])
    }

    /// Gets the span of the definition of the specified HIR node.
    /// This is used by `tcx.get_span`
    pub fn span(&self, hir_id: HirId) -> Span {
        match self.find_entry(hir_id).map(|entry| entry.node) {
            Some(Node::Param(param)) => param.span,
            Some(Node::Item(item)) => match &item.kind {
                ItemKind::Fn(sig, _, _) => sig.span,
                _ => item.span,
            },
            Some(Node::ForeignItem(foreign_item)) => foreign_item.span,
            Some(Node::TraitItem(trait_item)) => match &trait_item.kind {
                TraitItemKind::Fn(sig, _) => sig.span,
                _ => trait_item.span,
            },
            Some(Node::ImplItem(impl_item)) => match &impl_item.kind {
                ImplItemKind::Fn(sig, _) => sig.span,
                _ => impl_item.span,
            },
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
            Some(Node::Crate(item)) => item.span,
            None => bug!("hir::map::Map::span: id not in map: {:?}", hir_id),
        }
    }

    /// Like `hir.span()`, but includes the body of function items
    /// (instead of just the function header)
    pub fn span_with_body(&self, hir_id: HirId) -> Span {
        match self.find_entry(hir_id).map(|entry| entry.node) {
            Some(Node::TraitItem(item)) => item.span,
            Some(Node::ImplItem(impl_item)) => impl_item.span,
            Some(Node::Item(item)) => item.span,
            Some(_) => self.span(hir_id),
            _ => bug!("hir::map::Map::span_with_body: id not in map: {:?}", hir_id),
        }
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        id.as_local().map(|id| self.span(self.local_def_id_to_hir_id(id)))
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
}

impl<'hir> intravisit::Map<'hir> for Map<'hir> {
    fn find(&self, hir_id: HirId) -> Option<Node<'hir>> {
        self.find(hir_id)
    }

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

    fn foreign_item(&self, id: ForeignItemId) -> &'hir ForeignItem<'hir> {
        self.foreign_item(id)
    }
}

trait Named {
    fn name(&self) -> Symbol;
}

impl<T: Named> Named for Spanned<T> {
    fn name(&self) -> Symbol {
        self.node.name()
    }
}

impl Named for Item<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}
impl Named for ForeignItem<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}
impl Named for Variant<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}
impl Named for StructField<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}
impl Named for TraitItem<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}
impl Named for ImplItem<'_> {
    fn name(&self) -> Symbol {
        self.ident.name
    }
}

pub(super) fn index_hir<'tcx>(tcx: TyCtxt<'tcx>, cnum: CrateNum) -> &'tcx IndexedHir<'tcx> {
    assert_eq!(cnum, LOCAL_CRATE);

    let _prof_timer = tcx.sess.prof.generic_activity("build_hir_map");

    let (map, crate_hash) = {
        let hcx = tcx.create_stable_hashing_context();

        let mut collector =
            NodeCollector::root(tcx.sess, &**tcx.arena, tcx.untracked_crate, &tcx.definitions, hcx);
        intravisit::walk_crate(&mut collector, tcx.untracked_crate);

        let crate_disambiguator = tcx.sess.local_crate_disambiguator();
        let cmdline_args = tcx.sess.opts.dep_tracking_hash();
        collector.finalize_and_compute_crate_hash(crate_disambiguator, &*tcx.cstore, cmdline_args)
    };

    tcx.arena.alloc(IndexedHir { crate_hash, map })
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
        Some(Node::Local(_)) => node_str("local"),
        Some(Node::Ctor(..)) => format!("ctor {}{}", path_str(), id_str),
        Some(Node::Lifetime(_)) => node_str("lifetime"),
        Some(Node::GenericParam(ref param)) => format!("generic_param {:?}{}", param, id_str),
        Some(Node::Visibility(ref vis)) => format!("visibility {:?}{}", vis, id_str),
        Some(Node::MacroDef(_)) => format!("macro {}{}", path_str(), id_str),
        Some(Node::Crate(..)) => String::from("root_crate"),
        None => format!("unknown node{}", id_str),
    }
}

pub fn provide(providers: &mut Providers) {
    providers.def_kind = |tcx, def_id| tcx.hir().def_kind(def_id.expect_local());
}
