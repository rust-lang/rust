use rustc_ast::ptr::P;
use rustc_ast::visit::AssocCtxt;
use rustc_ast::*;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::PredicateOrigin;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::span_bug;
use rustc_middle::ty::{ResolverAstLowering, TyCtxt};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::symbol::{Ident, kw, sym};
use rustc_span::{DesugaringKind, Span, Symbol};
use rustc_target::spec::abi;
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;
use tracing::instrument;

use super::errors::{InvalidAbi, InvalidAbiReason, InvalidAbiSuggestion, MisplacedRelaxTraitBound};
use super::{
    AstOwner, FnDeclKind, ImplTraitContext, ImplTraitPosition, LoweringContext, ParamMode,
    ResolverAstLoweringExt,
};

pub(super) struct ItemLowerer<'a, 'hir> {
    pub(super) tcx: TyCtxt<'hir>,
    pub(super) resolver: &'a mut ResolverAstLowering,
    pub(super) ast_index: &'a IndexSlice<LocalDefId, AstOwner<'a>>,
    pub(super) owners: &'a mut IndexVec<LocalDefId, hir::MaybeOwner<'hir>>,
}

/// When we have a ty alias we *may* have two where clauses. To give the best diagnostics, we set the span
/// to the where clause that is preferred, if it exists. Otherwise, it sets the span to the other where
/// clause if it exists.
fn add_ty_alias_where_clause(
    generics: &mut ast::Generics,
    mut where_clauses: TyAliasWhereClauses,
    prefer_first: bool,
) {
    if !prefer_first {
        (where_clauses.before, where_clauses.after) = (where_clauses.after, where_clauses.before);
    }
    let where_clause =
        if where_clauses.before.has_where_token || !where_clauses.after.has_where_token {
            where_clauses.before
        } else {
            where_clauses.after
        };
    generics.where_clause.has_where_token = where_clause.has_where_token;
    generics.where_clause.span = where_clause.span;
}

impl<'a, 'hir> ItemLowerer<'a, 'hir> {
    fn with_lctx(
        &mut self,
        owner: NodeId,
        f: impl FnOnce(&mut LoweringContext<'_, 'hir>) -> hir::OwnerNode<'hir>,
    ) {
        let mut lctx = LoweringContext::new(self.tcx, self.resolver);
        lctx.with_hir_id_owner(owner, |lctx| f(lctx));

        for (def_id, info) in lctx.children {
            let owner = self.owners.ensure_contains_elem(def_id, || hir::MaybeOwner::Phantom);
            debug_assert!(
                matches!(owner, hir::MaybeOwner::Phantom),
                "duplicate copy of {def_id:?} in lctx.children"
            );
            *owner = info;
        }
    }

    pub(super) fn lower_node(&mut self, def_id: LocalDefId) -> hir::MaybeOwner<'hir> {
        let owner = self.owners.ensure_contains_elem(def_id, || hir::MaybeOwner::Phantom);
        if let hir::MaybeOwner::Phantom = owner {
            let node = self.ast_index[def_id];
            match node {
                AstOwner::NonOwner => {}
                AstOwner::Crate(c) => self.lower_crate(c),
                AstOwner::Item(item) => self.lower_item(item),
                AstOwner::AssocItem(item, ctxt) => self.lower_assoc_item(item, ctxt),
                AstOwner::ForeignItem(item) => self.lower_foreign_item(item),
            }
        }

        self.owners[def_id]
    }

    #[instrument(level = "debug", skip(self, c))]
    fn lower_crate(&mut self, c: &Crate) {
        debug_assert_eq!(self.resolver.node_id_to_def_id[&CRATE_NODE_ID], CRATE_DEF_ID);
        self.with_lctx(CRATE_NODE_ID, |lctx| {
            let module = lctx.lower_mod(&c.items, &c.spans);
            lctx.lower_attrs(hir::CRATE_HIR_ID, &c.attrs);
            hir::OwnerNode::Crate(module)
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_item(&mut self, item: &Item) {
        self.with_lctx(item.id, |lctx| hir::OwnerNode::Item(lctx.lower_item(item)))
    }

    fn lower_assoc_item(&mut self, item: &AssocItem, ctxt: AssocCtxt) {
        let def_id = self.resolver.node_id_to_def_id[&item.id];
        let parent_id = self.tcx.local_parent(def_id);
        let parent_hir = self.lower_node(parent_id).unwrap();
        self.with_lctx(item.id, |lctx| lctx.lower_assoc_item(item, ctxt, parent_hir))
    }

    fn lower_foreign_item(&mut self, item: &ForeignItem) {
        self.with_lctx(item.id, |lctx| hir::OwnerNode::ForeignItem(lctx.lower_foreign_item(item)))
    }
}

impl<'hir> LoweringContext<'_, 'hir> {
    pub(super) fn lower_mod(
        &mut self,
        items: &[P<Item>],
        spans: &ModSpans,
    ) -> &'hir hir::Mod<'hir> {
        self.arena.alloc(hir::Mod {
            spans: hir::ModSpans {
                inner_span: self.lower_span(spans.inner_span),
                inject_use_span: self.lower_span(spans.inject_use_span),
            },
            item_ids: self.arena.alloc_from_iter(items.iter().flat_map(|x| self.lower_item_ref(x))),
        })
    }

    pub(super) fn lower_item_ref(&mut self, i: &Item) -> SmallVec<[hir::ItemId; 1]> {
        let mut node_ids =
            smallvec![hir::ItemId { owner_id: hir::OwnerId { def_id: self.local_def_id(i.id) } }];
        if let ItemKind::Use(use_tree) = &i.kind {
            self.lower_item_id_use_tree(use_tree, &mut node_ids);
        }
        node_ids
    }

    fn lower_item_id_use_tree(&mut self, tree: &UseTree, vec: &mut SmallVec<[hir::ItemId; 1]>) {
        match &tree.kind {
            UseTreeKind::Nested { items, .. } => {
                for &(ref nested, id) in items {
                    vec.push(hir::ItemId {
                        owner_id: hir::OwnerId { def_id: self.local_def_id(id) },
                    });
                    self.lower_item_id_use_tree(nested, vec);
                }
            }
            UseTreeKind::Simple(..) | UseTreeKind::Glob => {}
        }
    }

    fn lower_item(&mut self, i: &Item) -> &'hir hir::Item<'hir> {
        let mut ident = i.ident;
        let vis_span = self.lower_span(i.vis.span);
        let hir_id = hir::HirId::make_owner(self.current_hir_id_owner.def_id);
        let attrs = self.lower_attrs(hir_id, &i.attrs);
        let kind = self.lower_item_kind(i.span, i.id, hir_id, &mut ident, attrs, vis_span, &i.kind);
        let item = hir::Item {
            owner_id: hir_id.expect_owner(),
            ident: self.lower_ident(ident),
            kind,
            vis_span,
            span: self.lower_span(i.span),
        };
        self.arena.alloc(item)
    }

    fn lower_item_kind(
        &mut self,
        span: Span,
        id: NodeId,
        hir_id: hir::HirId,
        ident: &mut Ident,
        attrs: &'hir [Attribute],
        vis_span: Span,
        i: &ItemKind,
    ) -> hir::ItemKind<'hir> {
        match i {
            ItemKind::ExternCrate(orig_name) => hir::ItemKind::ExternCrate(*orig_name),
            ItemKind::Use(use_tree) => {
                // Start with an empty prefix.
                let prefix = Path { segments: ThinVec::new(), span: use_tree.span, tokens: None };

                self.lower_use_tree(use_tree, &prefix, id, vis_span, ident, attrs)
            }
            ItemKind::Static(box ast::StaticItem { ty: t, safety: _, mutability: m, expr: e }) => {
                let (ty, body_id) =
                    self.lower_const_item(t, span, e.as_deref(), ImplTraitPosition::StaticTy);
                hir::ItemKind::Static(ty, *m, body_id)
            }
            ItemKind::Const(box ast::ConstItem { generics, ty, expr, .. }) => {
                let (generics, (ty, body_id)) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        this.lower_const_item(ty, span, expr.as_deref(), ImplTraitPosition::ConstTy)
                    },
                );
                hir::ItemKind::Const(ty, generics, body_id)
            }
            ItemKind::Fn(box Fn {
                sig: FnSig { decl, header, span: fn_sig_span },
                generics,
                body,
                ..
            }) => {
                self.with_new_scopes(*fn_sig_span, |this| {
                    // Note: we don't need to change the return type from `T` to
                    // `impl Future<Output = T>` here because lower_body
                    // only cares about the input argument patterns in the function
                    // declaration (decl), not the return types.
                    let coroutine_kind = header.coroutine_kind;
                    let body_id = this.lower_maybe_coroutine_body(
                        *fn_sig_span,
                        span,
                        hir_id,
                        decl,
                        coroutine_kind,
                        body.as_deref(),
                    );

                    let itctx = ImplTraitContext::Universal;
                    let (generics, decl) = this.lower_generics(generics, id, itctx, |this| {
                        this.lower_fn_decl(decl, id, *fn_sig_span, FnDeclKind::Fn, coroutine_kind)
                    });
                    let sig = hir::FnSig {
                        decl,
                        header: this.lower_fn_header(*header, hir::Safety::Safe),
                        span: this.lower_span(*fn_sig_span),
                    };
                    hir::ItemKind::Fn(sig, generics, body_id)
                })
            }
            ItemKind::Mod(_, mod_kind) => match mod_kind {
                ModKind::Loaded(items, _, spans) => {
                    hir::ItemKind::Mod(self.lower_mod(items, spans))
                }
                ModKind::Unloaded => panic!("`mod` items should have been loaded by now"),
            },
            ItemKind::ForeignMod(fm) => hir::ItemKind::ForeignMod {
                abi: fm.abi.map_or(abi::Abi::FALLBACK, |abi| self.lower_abi(abi)),
                items: self
                    .arena
                    .alloc_from_iter(fm.items.iter().map(|x| self.lower_foreign_item_ref(x))),
            },
            ItemKind::GlobalAsm(asm) => hir::ItemKind::GlobalAsm(self.lower_inline_asm(span, asm)),
            ItemKind::TyAlias(box TyAlias { generics, where_clauses, ty, .. }) => {
                // We lower
                //
                // type Foo = impl Trait
                //
                // to
                //
                // type Foo = Foo1
                // opaque type Foo1: Trait
                let mut generics = generics.clone();
                add_ty_alias_where_clause(&mut generics, *where_clauses, true);
                let (generics, ty) = self.lower_generics(
                    &generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| match ty {
                        None => {
                            let guar = this.dcx().span_delayed_bug(
                                span,
                                "expected to lower type alias type, but it was missing",
                            );
                            this.arena.alloc(this.ty(span, hir::TyKind::Err(guar)))
                        }
                        Some(ty) => this.lower_ty(ty, ImplTraitContext::OpaqueTy {
                            origin: hir::OpaqueTyOrigin::TyAlias {
                                parent: this.local_def_id(id),
                                in_assoc_ty: false,
                            },
                        }),
                    },
                );
                hir::ItemKind::TyAlias(ty, generics)
            }
            ItemKind::Enum(enum_definition, generics) => {
                let (generics, variants) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        this.arena.alloc_from_iter(
                            enum_definition.variants.iter().map(|x| this.lower_variant(x)),
                        )
                    },
                );
                hir::ItemKind::Enum(hir::EnumDef { variants }, generics)
            }
            ItemKind::Struct(struct_def, generics) => {
                let (generics, struct_def) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| this.lower_variant_data(hir_id, struct_def),
                );
                hir::ItemKind::Struct(struct_def, generics)
            }
            ItemKind::Union(vdata, generics) => {
                let (generics, vdata) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| this.lower_variant_data(hir_id, vdata),
                );
                hir::ItemKind::Union(vdata, generics)
            }
            ItemKind::Impl(box Impl {
                safety,
                polarity,
                defaultness,
                constness,
                generics: ast_generics,
                of_trait: trait_ref,
                self_ty: ty,
                items: impl_items,
            }) => {
                // Lower the "impl header" first. This ordering is important
                // for in-band lifetimes! Consider `'a` here:
                //
                //     impl Foo<'a> for u32 {
                //         fn method(&'a self) { .. }
                //     }
                //
                // Because we start by lowering the `Foo<'a> for u32`
                // part, we will add `'a` to the list of generics on
                // the impl. When we then encounter it later in the
                // method, it will not be considered an in-band
                // lifetime to be added, but rather a reference to a
                // parent lifetime.
                let itctx = ImplTraitContext::Universal;
                let (generics, (trait_ref, lowered_ty)) =
                    self.lower_generics(ast_generics, id, itctx, |this| {
                        let modifiers = TraitBoundModifiers {
                            constness: BoundConstness::Never,
                            asyncness: BoundAsyncness::Normal,
                            // we don't use this in bound lowering
                            polarity: BoundPolarity::Positive,
                        };

                        let trait_ref = trait_ref.as_ref().map(|trait_ref| {
                            this.lower_trait_ref(
                                modifiers,
                                trait_ref,
                                ImplTraitContext::Disallowed(ImplTraitPosition::Trait),
                            )
                        });

                        let lowered_ty = this.lower_ty(
                            ty,
                            ImplTraitContext::Disallowed(ImplTraitPosition::ImplSelf),
                        );

                        (trait_ref, lowered_ty)
                    });

                self.is_in_trait_impl = trait_ref.is_some();
                let new_impl_items = self
                    .arena
                    .alloc_from_iter(impl_items.iter().map(|item| self.lower_impl_item_ref(item)));

                // `defaultness.has_value()` is never called for an `impl`, always `true` in order
                // to not cause an assertion failure inside the `lower_defaultness` function.
                let has_val = true;
                let (defaultness, defaultness_span) = self.lower_defaultness(*defaultness, has_val);
                let polarity = match polarity {
                    ImplPolarity::Positive => ImplPolarity::Positive,
                    ImplPolarity::Negative(s) => ImplPolarity::Negative(self.lower_span(*s)),
                };
                hir::ItemKind::Impl(self.arena.alloc(hir::Impl {
                    constness: self.lower_constness(*constness),
                    safety: self.lower_safety(*safety, hir::Safety::Safe),
                    polarity,
                    defaultness,
                    defaultness_span,
                    generics,
                    of_trait: trait_ref,
                    self_ty: lowered_ty,
                    items: new_impl_items,
                }))
            }
            ItemKind::Trait(box Trait { is_auto, safety, generics, bounds, items }) => {
                let (generics, (safety, items, bounds)) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        let bounds = this.lower_param_bounds(
                            bounds,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                        );
                        let items = this.arena.alloc_from_iter(
                            items.iter().map(|item| this.lower_trait_item_ref(item)),
                        );
                        let safety = this.lower_safety(*safety, hir::Safety::Safe);
                        (safety, items, bounds)
                    },
                );
                hir::ItemKind::Trait(*is_auto, safety, generics, bounds, items)
            }
            ItemKind::TraitAlias(generics, bounds) => {
                let (generics, bounds) = self.lower_generics(
                    generics,
                    id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        this.lower_param_bounds(
                            bounds,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                        )
                    },
                );
                hir::ItemKind::TraitAlias(generics, bounds)
            }
            ItemKind::MacroDef(MacroDef { body, macro_rules }) => {
                let body = P(self.lower_delim_args(body));
                let def_id = self.local_def_id(id);
                let def_kind = self.tcx.def_kind(def_id);
                let DefKind::Macro(macro_kind) = def_kind else {
                    unreachable!(
                        "expected DefKind::Macro for macro item, found {}",
                        def_kind.descr(def_id.to_def_id())
                    );
                };
                let macro_def = self.arena.alloc(ast::MacroDef { body, macro_rules: *macro_rules });
                hir::ItemKind::Macro(macro_def, macro_kind)
            }
            ItemKind::Delegation(box delegation) => {
                let delegation_results = self.lower_delegation(delegation, id);
                hir::ItemKind::Fn(
                    delegation_results.sig,
                    delegation_results.generics,
                    delegation_results.body_id,
                )
            }
            ItemKind::MacCall(..) | ItemKind::DelegationMac(..) => {
                panic!("macros should have been expanded by now")
            }
        }
    }

    fn lower_const_item(
        &mut self,
        ty: &Ty,
        span: Span,
        body: Option<&Expr>,
        impl_trait_position: ImplTraitPosition,
    ) -> (&'hir hir::Ty<'hir>, hir::BodyId) {
        let ty = self.lower_ty(ty, ImplTraitContext::Disallowed(impl_trait_position));
        (ty, self.lower_const_body(span, body))
    }

    #[instrument(level = "debug", skip(self))]
    fn lower_use_tree(
        &mut self,
        tree: &UseTree,
        prefix: &Path,
        id: NodeId,
        vis_span: Span,
        ident: &mut Ident,
        attrs: &'hir [Attribute],
    ) -> hir::ItemKind<'hir> {
        let path = &tree.prefix;
        let segments = prefix.segments.iter().chain(path.segments.iter()).cloned().collect();

        match tree.kind {
            UseTreeKind::Simple(rename) => {
                *ident = tree.ident();

                // First, apply the prefix to the path.
                let mut path = Path { segments, span: path.span, tokens: None };

                // Correctly resolve `self` imports.
                if path.segments.len() > 1
                    && path.segments.last().unwrap().ident.name == kw::SelfLower
                {
                    let _ = path.segments.pop();
                    if rename.is_none() {
                        *ident = path.segments.last().unwrap().ident;
                    }
                }

                let res = self.lower_import_res(id, path.span);
                let path = self.lower_use_path(res, &path, ParamMode::Explicit);
                hir::ItemKind::Use(path, hir::UseKind::Single)
            }
            UseTreeKind::Glob => {
                let res = self.expect_full_res(id);
                let res = smallvec![self.lower_res(res)];
                let path = Path { segments, span: path.span, tokens: None };
                let path = self.lower_use_path(res, &path, ParamMode::Explicit);
                hir::ItemKind::Use(path, hir::UseKind::Glob)
            }
            UseTreeKind::Nested { items: ref trees, .. } => {
                // Nested imports are desugared into simple imports.
                // So, if we start with
                //
                // ```
                // pub(x) use foo::{a, b};
                // ```
                //
                // we will create three items:
                //
                // ```
                // pub(x) use foo::a;
                // pub(x) use foo::b;
                // pub(x) use foo::{}; // <-- this is called the `ListStem`
                // ```
                //
                // The first two are produced by recursively invoking
                // `lower_use_tree` (and indeed there may be things
                // like `use foo::{a::{b, c}}` and so forth). They
                // wind up being directly added to
                // `self.items`. However, the structure of this
                // function also requires us to return one item, and
                // for that we return the `{}` import (called the
                // `ListStem`).

                let span = prefix.span.to(path.span);
                let prefix = Path { segments, span, tokens: None };

                // Add all the nested `PathListItem`s to the HIR.
                for &(ref use_tree, id) in trees {
                    let new_hir_id = self.local_def_id(id);

                    // Each `use` import is an item and thus are owners of the
                    // names in the path. Up to this point the nested import is
                    // the current owner, since we want each desugared import to
                    // own its own names, we have to adjust the owner before
                    // lowering the rest of the import.
                    self.with_hir_id_owner(id, |this| {
                        let mut ident = *ident;

                        // `prefix` is lowered multiple times, but in different HIR owners.
                        // So each segment gets renewed `HirId` with the same
                        // `ItemLocalId` and the new owner. (See `lower_node_id`)
                        let kind =
                            this.lower_use_tree(use_tree, &prefix, id, vis_span, &mut ident, attrs);
                        if !attrs.is_empty() {
                            this.attrs.insert(hir::ItemLocalId::ZERO, attrs);
                        }

                        let item = hir::Item {
                            owner_id: hir::OwnerId { def_id: new_hir_id },
                            ident: this.lower_ident(ident),
                            kind,
                            vis_span,
                            span: this.lower_span(use_tree.span),
                        };
                        hir::OwnerNode::Item(this.arena.alloc(item))
                    });
                }

                // Condition should match `build_reduced_graph_for_use_tree`.
                let path = if trees.is_empty()
                    && !(prefix.segments.is_empty()
                        || prefix.segments.len() == 1
                            && prefix.segments[0].ident.name == kw::PathRoot)
                {
                    // For empty lists we need to lower the prefix so it is checked for things
                    // like stability later.
                    let res = self.lower_import_res(id, span);
                    self.lower_use_path(res, &prefix, ParamMode::Explicit)
                } else {
                    // For non-empty lists we can just drop all the data, the prefix is already
                    // present in HIR as a part of nested imports.
                    self.arena.alloc(hir::UsePath { res: smallvec![], segments: &[], span })
                };
                hir::ItemKind::Use(path, hir::UseKind::ListStem)
            }
        }
    }

    fn lower_assoc_item(
        &mut self,
        item: &AssocItem,
        ctxt: AssocCtxt,
        parent_hir: &'hir hir::OwnerInfo<'hir>,
    ) -> hir::OwnerNode<'hir> {
        let parent_item = parent_hir.node().expect_item();
        match parent_item.kind {
            hir::ItemKind::Impl(impl_) => {
                self.is_in_trait_impl = impl_.of_trait.is_some();
            }
            hir::ItemKind::Trait(_, _, _, _, _) => {}
            kind => {
                span_bug!(item.span, "assoc item has unexpected kind of parent: {}", kind.descr())
            }
        }

        // Evaluate with the lifetimes in `params` in-scope.
        // This is used to track which lifetimes have already been defined,
        // and which need to be replicated when lowering an async fn.
        match ctxt {
            AssocCtxt::Trait => hir::OwnerNode::TraitItem(self.lower_trait_item(item)),
            AssocCtxt::Impl => hir::OwnerNode::ImplItem(self.lower_impl_item(item)),
        }
    }

    fn lower_foreign_item(&mut self, i: &ForeignItem) -> &'hir hir::ForeignItem<'hir> {
        let hir_id = hir::HirId::make_owner(self.current_hir_id_owner.def_id);
        let owner_id = hir_id.expect_owner();
        self.lower_attrs(hir_id, &i.attrs);
        let item = hir::ForeignItem {
            owner_id,
            ident: self.lower_ident(i.ident),
            kind: match &i.kind {
                ForeignItemKind::Fn(box Fn { sig, generics, .. }) => {
                    let fdec = &sig.decl;
                    let itctx = ImplTraitContext::Universal;
                    let (generics, (decl, fn_args)) =
                        self.lower_generics(generics, i.id, itctx, |this| {
                            (
                                // Disallow `impl Trait` in foreign items.
                                this.lower_fn_decl(
                                    fdec,
                                    i.id,
                                    sig.span,
                                    FnDeclKind::ExternFn,
                                    None,
                                ),
                                this.lower_fn_params_to_names(fdec),
                            )
                        });

                    // Unmarked safety in unsafe block defaults to unsafe.
                    let header = self.lower_fn_header(sig.header, hir::Safety::Unsafe);

                    hir::ForeignItemKind::Fn(
                        hir::FnSig { header, decl, span: self.lower_span(sig.span) },
                        fn_args,
                        generics,
                    )
                }
                ForeignItemKind::Static(box StaticItem { ty, mutability, expr: _, safety }) => {
                    let ty = self
                        .lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::StaticTy));
                    let safety = self.lower_safety(*safety, hir::Safety::Unsafe);

                    hir::ForeignItemKind::Static(ty, *mutability, safety)
                }
                ForeignItemKind::TyAlias(..) => hir::ForeignItemKind::Type,
                ForeignItemKind::MacCall(_) => panic!("macro shouldn't exist here"),
            },
            vis_span: self.lower_span(i.vis.span),
            span: self.lower_span(i.span),
        };
        self.arena.alloc(item)
    }

    fn lower_foreign_item_ref(&mut self, i: &ForeignItem) -> hir::ForeignItemRef {
        hir::ForeignItemRef {
            id: hir::ForeignItemId { owner_id: hir::OwnerId { def_id: self.local_def_id(i.id) } },
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
        }
    }

    fn lower_variant(&mut self, v: &Variant) -> hir::Variant<'hir> {
        let hir_id = self.lower_node_id(v.id);
        self.lower_attrs(hir_id, &v.attrs);
        hir::Variant {
            hir_id,
            def_id: self.local_def_id(v.id),
            data: self.lower_variant_data(hir_id, &v.data),
            disr_expr: v.disr_expr.as_ref().map(|e| self.lower_anon_const_to_anon_const(e)),
            ident: self.lower_ident(v.ident),
            span: self.lower_span(v.span),
        }
    }

    fn lower_variant_data(
        &mut self,
        parent_id: hir::HirId,
        vdata: &VariantData,
    ) -> hir::VariantData<'hir> {
        match vdata {
            VariantData::Struct { fields, recovered } => hir::VariantData::Struct {
                fields: self
                    .arena
                    .alloc_from_iter(fields.iter().enumerate().map(|f| self.lower_field_def(f))),
                recovered: *recovered,
            },
            VariantData::Tuple(fields, id) => {
                let ctor_id = self.lower_node_id(*id);
                self.alias_attrs(ctor_id, parent_id);
                hir::VariantData::Tuple(
                    self.arena.alloc_from_iter(
                        fields.iter().enumerate().map(|f| self.lower_field_def(f)),
                    ),
                    ctor_id,
                    self.local_def_id(*id),
                )
            }
            VariantData::Unit(id) => {
                let ctor_id = self.lower_node_id(*id);
                self.alias_attrs(ctor_id, parent_id);
                hir::VariantData::Unit(ctor_id, self.local_def_id(*id))
            }
        }
    }

    pub(super) fn lower_field_def(
        &mut self,
        (index, f): (usize, &FieldDef),
    ) -> hir::FieldDef<'hir> {
        let ty = self.lower_ty(&f.ty, ImplTraitContext::Disallowed(ImplTraitPosition::FieldTy));
        let hir_id = self.lower_node_id(f.id);
        self.lower_attrs(hir_id, &f.attrs);
        hir::FieldDef {
            span: self.lower_span(f.span),
            hir_id,
            def_id: self.local_def_id(f.id),
            ident: match f.ident {
                Some(ident) => self.lower_ident(ident),
                // FIXME(jseyfried): positional field hygiene.
                None => Ident::new(sym::integer(index), self.lower_span(f.span)),
            },
            vis_span: self.lower_span(f.vis.span),
            ty,
            safety: self.lower_safety(f.safety, hir::Safety::Safe),
        }
    }

    fn lower_trait_item(&mut self, i: &AssocItem) -> &'hir hir::TraitItem<'hir> {
        let hir_id = hir::HirId::make_owner(self.current_hir_id_owner.def_id);
        self.lower_attrs(hir_id, &i.attrs);
        let trait_item_def_id = hir_id.expect_owner();

        let (generics, kind, has_default) = match &i.kind {
            AssocItemKind::Const(box ConstItem { generics, ty, expr, .. }) => {
                let (generics, kind) = self.lower_generics(
                    generics,
                    i.id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        let ty = this
                            .lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::ConstTy));
                        let body = expr.as_ref().map(|x| this.lower_const_body(i.span, Some(x)));

                        hir::TraitItemKind::Const(ty, body)
                    },
                );
                (generics, kind, expr.is_some())
            }
            AssocItemKind::Fn(box Fn { sig, generics, body: None, .. }) => {
                let names = self.lower_fn_params_to_names(&sig.decl);
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    i.id,
                    FnDeclKind::Trait,
                    sig.header.coroutine_kind,
                );
                (generics, hir::TraitItemKind::Fn(sig, hir::TraitFn::Required(names)), false)
            }
            AssocItemKind::Fn(box Fn { sig, generics, body: Some(body), .. }) => {
                let body_id = self.lower_maybe_coroutine_body(
                    sig.span,
                    i.span,
                    hir_id,
                    &sig.decl,
                    sig.header.coroutine_kind,
                    Some(body),
                );
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    i.id,
                    FnDeclKind::Trait,
                    sig.header.coroutine_kind,
                );
                (generics, hir::TraitItemKind::Fn(sig, hir::TraitFn::Provided(body_id)), true)
            }
            AssocItemKind::Type(box TyAlias { generics, where_clauses, bounds, ty, .. }) => {
                let mut generics = generics.clone();
                add_ty_alias_where_clause(&mut generics, *where_clauses, false);
                let (generics, kind) = self.lower_generics(
                    &generics,
                    i.id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| {
                        let ty = ty.as_ref().map(|x| {
                            this.lower_ty(
                                x,
                                ImplTraitContext::Disallowed(ImplTraitPosition::AssocTy),
                            )
                        });
                        hir::TraitItemKind::Type(
                            this.lower_param_bounds(
                                bounds,
                                ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                            ),
                            ty,
                        )
                    },
                );
                (generics, kind, ty.is_some())
            }
            AssocItemKind::Delegation(box delegation) => {
                let delegation_results = self.lower_delegation(delegation, i.id);
                let item_kind = hir::TraitItemKind::Fn(
                    delegation_results.sig,
                    hir::TraitFn::Provided(delegation_results.body_id),
                );
                (delegation_results.generics, item_kind, true)
            }
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                panic!("macros should have been expanded by now")
            }
        };

        let item = hir::TraitItem {
            owner_id: trait_item_def_id,
            ident: self.lower_ident(i.ident),
            generics,
            kind,
            span: self.lower_span(i.span),
            defaultness: hir::Defaultness::Default { has_value: has_default },
        };
        self.arena.alloc(item)
    }

    fn lower_trait_item_ref(&mut self, i: &AssocItem) -> hir::TraitItemRef {
        let kind = match &i.kind {
            AssocItemKind::Const(..) => hir::AssocItemKind::Const,
            AssocItemKind::Type(..) => hir::AssocItemKind::Type,
            AssocItemKind::Fn(box Fn { sig, .. }) => {
                hir::AssocItemKind::Fn { has_self: sig.decl.has_self() }
            }
            AssocItemKind::Delegation(box delegation) => hir::AssocItemKind::Fn {
                has_self: self.delegation_has_self(i.id, delegation.id, i.span),
            },
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                panic!("macros should have been expanded by now")
            }
        };
        let id = hir::TraitItemId { owner_id: hir::OwnerId { def_id: self.local_def_id(i.id) } };
        hir::TraitItemRef {
            id,
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
            kind,
        }
    }

    /// Construct `ExprKind::Err` for the given `span`.
    pub(crate) fn expr_err(&mut self, span: Span, guar: ErrorGuaranteed) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::Err(guar))
    }

    fn lower_impl_item(&mut self, i: &AssocItem) -> &'hir hir::ImplItem<'hir> {
        // Since `default impl` is not yet implemented, this is always true in impls.
        let has_value = true;
        let (defaultness, _) = self.lower_defaultness(i.kind.defaultness(), has_value);
        let hir_id = hir::HirId::make_owner(self.current_hir_id_owner.def_id);
        self.lower_attrs(hir_id, &i.attrs);

        let (generics, kind) = match &i.kind {
            AssocItemKind::Const(box ConstItem { generics, ty, expr, .. }) => self.lower_generics(
                generics,
                i.id,
                ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                |this| {
                    let ty =
                        this.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::ConstTy));
                    let body = this.lower_const_body(i.span, expr.as_deref());

                    hir::ImplItemKind::Const(ty, body)
                },
            ),
            AssocItemKind::Fn(box Fn { sig, generics, body, .. }) => {
                let body_id = self.lower_maybe_coroutine_body(
                    sig.span,
                    i.span,
                    hir_id,
                    &sig.decl,
                    sig.header.coroutine_kind,
                    body.as_deref(),
                );
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    i.id,
                    if self.is_in_trait_impl { FnDeclKind::Impl } else { FnDeclKind::Inherent },
                    sig.header.coroutine_kind,
                );

                (generics, hir::ImplItemKind::Fn(sig, body_id))
            }
            AssocItemKind::Type(box TyAlias { generics, where_clauses, ty, .. }) => {
                let mut generics = generics.clone();
                add_ty_alias_where_clause(&mut generics, *where_clauses, false);
                self.lower_generics(
                    &generics,
                    i.id,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    |this| match ty {
                        None => {
                            let guar = this.dcx().span_delayed_bug(
                                i.span,
                                "expected to lower associated type, but it was missing",
                            );
                            let ty = this.arena.alloc(this.ty(i.span, hir::TyKind::Err(guar)));
                            hir::ImplItemKind::Type(ty)
                        }
                        Some(ty) => {
                            let ty = this.lower_ty(ty, ImplTraitContext::OpaqueTy {
                                origin: hir::OpaqueTyOrigin::TyAlias {
                                    parent: this.local_def_id(i.id),
                                    in_assoc_ty: true,
                                },
                            });
                            hir::ImplItemKind::Type(ty)
                        }
                    },
                )
            }
            AssocItemKind::Delegation(box delegation) => {
                let delegation_results = self.lower_delegation(delegation, i.id);
                (
                    delegation_results.generics,
                    hir::ImplItemKind::Fn(delegation_results.sig, delegation_results.body_id),
                )
            }
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                panic!("macros should have been expanded by now")
            }
        };

        let item = hir::ImplItem {
            owner_id: hir_id.expect_owner(),
            ident: self.lower_ident(i.ident),
            generics,
            kind,
            vis_span: self.lower_span(i.vis.span),
            span: self.lower_span(i.span),
            defaultness,
        };
        self.arena.alloc(item)
    }

    fn lower_impl_item_ref(&mut self, i: &AssocItem) -> hir::ImplItemRef {
        hir::ImplItemRef {
            id: hir::ImplItemId { owner_id: hir::OwnerId { def_id: self.local_def_id(i.id) } },
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
            kind: match &i.kind {
                AssocItemKind::Const(..) => hir::AssocItemKind::Const,
                AssocItemKind::Type(..) => hir::AssocItemKind::Type,
                AssocItemKind::Fn(box Fn { sig, .. }) => {
                    hir::AssocItemKind::Fn { has_self: sig.decl.has_self() }
                }
                AssocItemKind::Delegation(box delegation) => hir::AssocItemKind::Fn {
                    has_self: self.delegation_has_self(i.id, delegation.id, i.span),
                },
                AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                    panic!("macros should have been expanded by now")
                }
            },
            trait_item_def_id: self
                .resolver
                .get_partial_res(i.id)
                .map(|r| r.expect_full_res().opt_def_id())
                .unwrap_or(None),
        }
    }

    fn lower_defaultness(
        &self,
        d: Defaultness,
        has_value: bool,
    ) -> (hir::Defaultness, Option<Span>) {
        match d {
            Defaultness::Default(sp) => {
                (hir::Defaultness::Default { has_value }, Some(self.lower_span(sp)))
            }
            Defaultness::Final => {
                assert!(has_value);
                (hir::Defaultness::Final, None)
            }
        }
    }

    fn record_body(
        &mut self,
        params: &'hir [hir::Param<'hir>],
        value: hir::Expr<'hir>,
    ) -> hir::BodyId {
        let body = hir::Body { params, value: self.arena.alloc(value) };
        let id = body.id();
        debug_assert_eq!(id.hir_id.owner, self.current_hir_id_owner);
        self.bodies.push((id.hir_id.local_id, self.arena.alloc(body)));
        id
    }

    pub(super) fn lower_body(
        &mut self,
        f: impl FnOnce(&mut Self) -> (&'hir [hir::Param<'hir>], hir::Expr<'hir>),
    ) -> hir::BodyId {
        let prev_coroutine_kind = self.coroutine_kind.take();
        let task_context = self.task_context.take();
        let (parameters, result) = f(self);
        let body_id = self.record_body(parameters, result);
        self.task_context = task_context;
        self.coroutine_kind = prev_coroutine_kind;
        body_id
    }

    fn lower_param(&mut self, param: &Param) -> hir::Param<'hir> {
        let hir_id = self.lower_node_id(param.id);
        self.lower_attrs(hir_id, &param.attrs);
        hir::Param {
            hir_id,
            pat: self.lower_pat(&param.pat),
            ty_span: self.lower_span(param.ty.span),
            span: self.lower_span(param.span),
        }
    }

    pub(super) fn lower_fn_body(
        &mut self,
        decl: &FnDecl,
        body: impl FnOnce(&mut Self) -> hir::Expr<'hir>,
    ) -> hir::BodyId {
        self.lower_body(|this| {
            (
                this.arena.alloc_from_iter(decl.inputs.iter().map(|x| this.lower_param(x))),
                body(this),
            )
        })
    }

    fn lower_fn_body_block(
        &mut self,
        span: Span,
        decl: &FnDecl,
        body: Option<&Block>,
    ) -> hir::BodyId {
        self.lower_fn_body(decl, |this| this.lower_block_expr_opt(span, body))
    }

    fn lower_block_expr_opt(&mut self, span: Span, block: Option<&Block>) -> hir::Expr<'hir> {
        match block {
            Some(block) => self.lower_block_expr(block),
            None => self.expr_err(span, self.dcx().has_errors().unwrap()),
        }
    }

    pub(super) fn lower_const_body(&mut self, span: Span, expr: Option<&Expr>) -> hir::BodyId {
        self.lower_body(|this| {
            (&[], match expr {
                Some(expr) => this.lower_expr_mut(expr),
                None => this.expr_err(span, this.dcx().span_delayed_bug(span, "no block")),
            })
        })
    }

    /// Takes what may be the body of an `async fn` or a `gen fn` and wraps it in an `async {}` or
    /// `gen {}` block as appropriate.
    fn lower_maybe_coroutine_body(
        &mut self,
        fn_decl_span: Span,
        span: Span,
        fn_id: hir::HirId,
        decl: &FnDecl,
        coroutine_kind: Option<CoroutineKind>,
        body: Option<&Block>,
    ) -> hir::BodyId {
        let (Some(coroutine_kind), Some(body)) = (coroutine_kind, body) else {
            return self.lower_fn_body_block(span, decl, body);
        };
        self.lower_body(|this| {
            let (parameters, expr) = this.lower_coroutine_body_with_moved_arguments(
                decl,
                |this| this.lower_block_expr(body),
                fn_decl_span,
                body.span,
                coroutine_kind,
                hir::CoroutineSource::Fn,
            );

            // FIXME(async_fn_track_caller): Can this be moved above?
            let hir_id = expr.hir_id;
            this.maybe_forward_track_caller(body.span, fn_id, hir_id);

            (parameters, expr)
        })
    }

    /// Lowers a desugared coroutine body after moving all of the arguments
    /// into the body. This is to make sure that the future actually owns the
    /// arguments that are passed to the function, and to ensure things like
    /// drop order are stable.
    pub(crate) fn lower_coroutine_body_with_moved_arguments(
        &mut self,
        decl: &FnDecl,
        lower_body: impl FnOnce(&mut LoweringContext<'_, 'hir>) -> hir::Expr<'hir>,
        fn_decl_span: Span,
        body_span: Span,
        coroutine_kind: CoroutineKind,
        coroutine_source: hir::CoroutineSource,
    ) -> (&'hir [hir::Param<'hir>], hir::Expr<'hir>) {
        let mut parameters: Vec<hir::Param<'_>> = Vec::new();
        let mut statements: Vec<hir::Stmt<'_>> = Vec::new();

        // Async function parameters are lowered into the closure body so that they are
        // captured and so that the drop order matches the equivalent non-async functions.
        //
        // from:
        //
        //     async fn foo(<pattern>: <ty>, <pattern>: <ty>, <pattern>: <ty>) {
        //         <body>
        //     }
        //
        // into:
        //
        //     fn foo(__arg0: <ty>, __arg1: <ty>, __arg2: <ty>) {
        //       async move {
        //         let __arg2 = __arg2;
        //         let <pattern> = __arg2;
        //         let __arg1 = __arg1;
        //         let <pattern> = __arg1;
        //         let __arg0 = __arg0;
        //         let <pattern> = __arg0;
        //         drop-temps { <body> } // see comments later in fn for details
        //       }
        //     }
        //
        // If `<pattern>` is a simple ident, then it is lowered to a single
        // `let <pattern> = <pattern>;` statement as an optimization.
        //
        // Note that the body is embedded in `drop-temps`; an
        // equivalent desugaring would be `return { <body>
        // };`. The key point is that we wish to drop all the
        // let-bound variables and temporaries created in the body
        // (and its tail expression!) before we drop the
        // parameters (c.f. rust-lang/rust#64512).
        for (index, parameter) in decl.inputs.iter().enumerate() {
            let parameter = self.lower_param(parameter);
            let span = parameter.pat.span;

            // Check if this is a binding pattern, if so, we can optimize and avoid adding a
            // `let <pat> = __argN;` statement. In this case, we do not rename the parameter.
            let (ident, is_simple_parameter) = match parameter.pat.kind {
                hir::PatKind::Binding(hir::BindingMode(ByRef::No, _), _, ident, _) => (ident, true),
                // For `ref mut` or wildcard arguments, we can't reuse the binding, but
                // we can keep the same name for the parameter.
                // This lets rustdoc render it correctly in documentation.
                hir::PatKind::Binding(_, _, ident, _) => (ident, false),
                hir::PatKind::Wild => {
                    (Ident::with_dummy_span(rustc_span::symbol::kw::Underscore), false)
                }
                _ => {
                    // Replace the ident for bindings that aren't simple.
                    let name = format!("__arg{index}");
                    let ident = Ident::from_str(&name);

                    (ident, false)
                }
            };

            let desugared_span = self.mark_span_with_reason(DesugaringKind::Async, span, None);

            // Construct a parameter representing `__argN: <ty>` to replace the parameter of the
            // async function.
            //
            // If this is the simple case, this parameter will end up being the same as the
            // original parameter, but with a different pattern id.
            let stmt_attrs = self.attrs.get(&parameter.hir_id.local_id).copied();
            let (new_parameter_pat, new_parameter_id) = self.pat_ident(desugared_span, ident);
            let new_parameter = hir::Param {
                hir_id: parameter.hir_id,
                pat: new_parameter_pat,
                ty_span: self.lower_span(parameter.ty_span),
                span: self.lower_span(parameter.span),
            };

            if is_simple_parameter {
                // If this is the simple case, then we only insert one statement that is
                // `let <pat> = <pat>;`. We re-use the original argument's pattern so that
                // `HirId`s are densely assigned.
                let expr = self.expr_ident(desugared_span, ident, new_parameter_id);
                let stmt = self.stmt_let_pat(
                    stmt_attrs,
                    desugared_span,
                    Some(expr),
                    parameter.pat,
                    hir::LocalSource::AsyncFn,
                );
                statements.push(stmt);
            } else {
                // If this is not the simple case, then we construct two statements:
                //
                // ```
                // let __argN = __argN;
                // let <pat> = __argN;
                // ```
                //
                // The first statement moves the parameter into the closure and thus ensures
                // that the drop order is correct.
                //
                // The second statement creates the bindings that the user wrote.

                // Construct the `let mut __argN = __argN;` statement. It must be a mut binding
                // because the user may have specified a `ref mut` binding in the next
                // statement.
                let (move_pat, move_id) =
                    self.pat_ident_binding_mode(desugared_span, ident, hir::BindingMode::MUT);
                let move_expr = self.expr_ident(desugared_span, ident, new_parameter_id);
                let move_stmt = self.stmt_let_pat(
                    None,
                    desugared_span,
                    Some(move_expr),
                    move_pat,
                    hir::LocalSource::AsyncFn,
                );

                // Construct the `let <pat> = __argN;` statement. We re-use the original
                // parameter's pattern so that `HirId`s are densely assigned.
                let pattern_expr = self.expr_ident(desugared_span, ident, move_id);
                let pattern_stmt = self.stmt_let_pat(
                    stmt_attrs,
                    desugared_span,
                    Some(pattern_expr),
                    parameter.pat,
                    hir::LocalSource::AsyncFn,
                );

                statements.push(move_stmt);
                statements.push(pattern_stmt);
            };

            parameters.push(new_parameter);
        }

        let mkbody = |this: &mut LoweringContext<'_, 'hir>| {
            // Create a block from the user's function body:
            let user_body = lower_body(this);

            // Transform into `drop-temps { <user-body> }`, an expression:
            let desugared_span =
                this.mark_span_with_reason(DesugaringKind::Async, user_body.span, None);
            let user_body = this.expr_drop_temps(desugared_span, this.arena.alloc(user_body));

            // As noted above, create the final block like
            //
            // ```
            // {
            //   let $param_pattern = $raw_param;
            //   ...
            //   drop-temps { <user-body> }
            // }
            // ```
            let body = this.block_all(
                desugared_span,
                this.arena.alloc_from_iter(statements),
                Some(user_body),
            );

            this.expr_block(body)
        };
        let desugaring_kind = match coroutine_kind {
            CoroutineKind::Async { .. } => hir::CoroutineDesugaring::Async,
            CoroutineKind::Gen { .. } => hir::CoroutineDesugaring::Gen,
            CoroutineKind::AsyncGen { .. } => hir::CoroutineDesugaring::AsyncGen,
        };
        let closure_id = coroutine_kind.closure_id();

        let coroutine_expr = self.make_desugared_coroutine_expr(
            // The default capture mode here is by-ref. Later on during upvar analysis,
            // we will force the captured arguments to by-move, but for async closures,
            // we want to make sure that we avoid unnecessarily moving captures, or else
            // all async closures would default to `FnOnce` as their calling mode.
            CaptureBy::Ref,
            closure_id,
            None,
            fn_decl_span,
            body_span,
            desugaring_kind,
            coroutine_source,
            mkbody,
        );

        let expr = hir::Expr {
            hir_id: self.lower_node_id(closure_id),
            kind: coroutine_expr,
            span: self.lower_span(body_span),
        };

        (self.arena.alloc_from_iter(parameters), expr)
    }

    fn lower_method_sig(
        &mut self,
        generics: &Generics,
        sig: &FnSig,
        id: NodeId,
        kind: FnDeclKind,
        coroutine_kind: Option<CoroutineKind>,
    ) -> (&'hir hir::Generics<'hir>, hir::FnSig<'hir>) {
        let header = self.lower_fn_header(sig.header, hir::Safety::Safe);
        let itctx = ImplTraitContext::Universal;
        let (generics, decl) = self.lower_generics(generics, id, itctx, |this| {
            this.lower_fn_decl(&sig.decl, id, sig.span, kind, coroutine_kind)
        });
        (generics, hir::FnSig { header, decl, span: self.lower_span(sig.span) })
    }

    pub(super) fn lower_fn_header(
        &mut self,
        h: FnHeader,
        default_safety: hir::Safety,
    ) -> hir::FnHeader {
        let asyncness = if let Some(CoroutineKind::Async { span, .. }) = h.coroutine_kind {
            hir::IsAsync::Async(span)
        } else {
            hir::IsAsync::NotAsync
        };
        hir::FnHeader {
            safety: self.lower_safety(h.safety, default_safety),
            asyncness,
            constness: self.lower_constness(h.constness),
            abi: self.lower_extern(h.ext),
        }
    }

    pub(super) fn lower_abi(&mut self, abi: StrLit) -> abi::Abi {
        abi::lookup(abi.symbol_unescaped.as_str()).unwrap_or_else(|err| {
            self.error_on_invalid_abi(abi, err);
            abi::Abi::Rust
        })
    }

    pub(super) fn lower_extern(&mut self, ext: Extern) -> abi::Abi {
        match ext {
            Extern::None => abi::Abi::Rust,
            Extern::Implicit(_) => abi::Abi::FALLBACK,
            Extern::Explicit(abi, _) => self.lower_abi(abi),
        }
    }

    fn error_on_invalid_abi(&self, abi: StrLit, err: abi::AbiUnsupported) {
        let abi_names = abi::enabled_names(self.tcx.features(), abi.span)
            .iter()
            .map(|s| Symbol::intern(s))
            .collect::<Vec<_>>();
        let suggested_name = find_best_match_for_name(&abi_names, abi.symbol_unescaped, None);
        self.dcx().emit_err(InvalidAbi {
            abi: abi.symbol_unescaped,
            span: abi.span,
            explain: match err {
                abi::AbiUnsupported::Reason { explain } => Some(InvalidAbiReason(explain)),
                _ => None,
            },
            suggestion: suggested_name.map(|suggested_name| InvalidAbiSuggestion {
                span: abi.span,
                suggestion: format!("\"{suggested_name}\""),
            }),
            command: "rustc --print=calling-conventions".to_string(),
        });
    }

    pub(super) fn lower_constness(&mut self, c: Const) -> hir::Constness {
        match c {
            Const::Yes(_) => hir::Constness::Const,
            Const::No => hir::Constness::NotConst,
        }
    }

    pub(super) fn lower_safety(&mut self, s: Safety, default: hir::Safety) -> hir::Safety {
        match s {
            Safety::Unsafe(_) => hir::Safety::Unsafe,
            Safety::Default => default,
            Safety::Safe(_) => hir::Safety::Safe,
        }
    }

    /// Return the pair of the lowered `generics` as `hir::Generics` and the evaluation of `f` with
    /// the carried impl trait definitions and bounds.
    #[instrument(level = "debug", skip(self, f))]
    fn lower_generics<T>(
        &mut self,
        generics: &Generics,
        parent_node_id: NodeId,
        itctx: ImplTraitContext,
        f: impl FnOnce(&mut Self) -> T,
    ) -> (&'hir hir::Generics<'hir>, T) {
        debug_assert!(self.impl_trait_defs.is_empty());
        debug_assert!(self.impl_trait_bounds.is_empty());

        // Error if `?Trait` bounds in where clauses don't refer directly to type parameters.
        // Note: we used to clone these bounds directly onto the type parameter (and avoid lowering
        // these into hir when we lower thee where clauses), but this makes it quite difficult to
        // keep track of the Span info. Now, `<dyn HirTyLowerer>::add_implicit_sized_bound`
        // checks both param bounds and where clauses for `?Sized`.
        for pred in &generics.where_clause.predicates {
            let WherePredicateKind::BoundPredicate(bound_pred) = &pred.kind else {
                continue;
            };
            let compute_is_param = || {
                // Check if the where clause type is a plain type parameter.
                match self
                    .resolver
                    .get_partial_res(bound_pred.bounded_ty.id)
                    .and_then(|r| r.full_res())
                {
                    Some(Res::Def(DefKind::TyParam, def_id))
                        if bound_pred.bound_generic_params.is_empty() =>
                    {
                        generics
                            .params
                            .iter()
                            .any(|p| def_id == self.local_def_id(p.id).to_def_id())
                    }
                    // Either the `bounded_ty` is not a plain type parameter, or
                    // it's not found in the generic type parameters list.
                    _ => false,
                }
            };
            // We only need to compute this once per `WherePredicate`, but don't
            // need to compute this at all unless there is a Maybe bound.
            let mut is_param: Option<bool> = None;
            for bound in &bound_pred.bounds {
                if !matches!(
                    *bound,
                    GenericBound::Trait(PolyTraitRef {
                        modifiers: TraitBoundModifiers { polarity: BoundPolarity::Maybe(_), .. },
                        ..
                    })
                ) {
                    continue;
                }
                let is_param = *is_param.get_or_insert_with(compute_is_param);
                if !is_param && !self.tcx.features().more_maybe_bounds() {
                    self.tcx
                        .sess
                        .create_feature_err(
                            MisplacedRelaxTraitBound { span: bound.span() },
                            sym::more_maybe_bounds,
                        )
                        .emit();
                }
            }
        }

        let mut predicates: SmallVec<[hir::WherePredicate<'hir>; 4]> = SmallVec::new();
        predicates.extend(generics.params.iter().filter_map(|param| {
            self.lower_generic_bound_predicate(
                param.ident,
                param.id,
                &param.kind,
                &param.bounds,
                param.colon_span,
                generics.span,
                itctx,
                PredicateOrigin::GenericParam,
            )
        }));
        predicates.extend(
            generics
                .where_clause
                .predicates
                .iter()
                .map(|predicate| self.lower_where_predicate(predicate)),
        );

        let mut params: SmallVec<[hir::GenericParam<'hir>; 4]> = self
            .lower_generic_params_mut(&generics.params, hir::GenericParamSource::Generics)
            .collect();

        // Introduce extra lifetimes if late resolution tells us to.
        let extra_lifetimes = self.resolver.extra_lifetime_params(parent_node_id);
        params.extend(extra_lifetimes.into_iter().filter_map(|(ident, node_id, res)| {
            self.lifetime_res_to_generic_param(
                ident,
                node_id,
                res,
                hir::GenericParamSource::Generics,
            )
        }));

        let has_where_clause_predicates = !generics.where_clause.predicates.is_empty();
        let where_clause_span = self.lower_span(generics.where_clause.span);
        let span = self.lower_span(generics.span);
        let res = f(self);

        let impl_trait_defs = std::mem::take(&mut self.impl_trait_defs);
        params.extend(impl_trait_defs.into_iter());

        let impl_trait_bounds = std::mem::take(&mut self.impl_trait_bounds);
        predicates.extend(impl_trait_bounds.into_iter());

        let lowered_generics = self.arena.alloc(hir::Generics {
            params: self.arena.alloc_from_iter(params),
            predicates: self.arena.alloc_from_iter(predicates),
            has_where_clause_predicates,
            where_clause_span,
            span,
        });

        (lowered_generics, res)
    }

    pub(super) fn lower_generic_bound_predicate(
        &mut self,
        ident: Ident,
        id: NodeId,
        kind: &GenericParamKind,
        bounds: &[GenericBound],
        colon_span: Option<Span>,
        parent_span: Span,
        itctx: ImplTraitContext,
        origin: PredicateOrigin,
    ) -> Option<hir::WherePredicate<'hir>> {
        // Do not create a clause if we do not have anything inside it.
        if bounds.is_empty() {
            return None;
        }

        let bounds = self.lower_param_bounds(bounds, itctx);

        let ident = self.lower_ident(ident);
        let param_span = ident.span;

        // Reconstruct the span of the entire predicate from the individual generic bounds.
        let span_start = colon_span.unwrap_or_else(|| param_span.shrink_to_hi());
        let span = bounds.iter().fold(span_start, |span_accum, bound| {
            match bound.span().find_ancestor_inside(parent_span) {
                Some(bound_span) => span_accum.to(bound_span),
                None => span_accum,
            }
        });
        let span = self.lower_span(span);
        let hir_id = self.next_id();
        let kind = self.arena.alloc(match kind {
            GenericParamKind::Const { .. } => return None,
            GenericParamKind::Type { .. } => {
                let def_id = self.local_def_id(id).to_def_id();
                let hir_id = self.next_id();
                let res = Res::Def(DefKind::TyParam, def_id);
                let ty_path = self.arena.alloc(hir::Path {
                    span: param_span,
                    res,
                    segments: self
                        .arena
                        .alloc_from_iter([hir::PathSegment::new(ident, hir_id, res)]),
                });
                let ty_id = self.next_id();
                let bounded_ty =
                    self.ty_path(ty_id, param_span, hir::QPath::Resolved(None, ty_path));
                hir::WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                    bounded_ty: self.arena.alloc(bounded_ty),
                    bounds,
                    bound_generic_params: &[],
                    origin,
                })
            }
            GenericParamKind::Lifetime => {
                let ident = self.lower_ident(ident);
                let lt_id = self.next_node_id();
                let lifetime = self.new_named_lifetime(id, lt_id, ident);
                hir::WherePredicateKind::RegionPredicate(hir::WhereRegionPredicate {
                    lifetime,
                    bounds,
                    in_where_clause: false,
                })
            }
        });
        Some(hir::WherePredicate { hir_id, span, kind })
    }

    fn lower_where_predicate(&mut self, pred: &WherePredicate) -> hir::WherePredicate<'hir> {
        let hir_id = self.lower_node_id(pred.id);
        let span = self.lower_span(pred.span);
        let kind = self.arena.alloc(match &pred.kind {
            WherePredicateKind::BoundPredicate(WhereBoundPredicate {
                bound_generic_params,
                bounded_ty,
                bounds,
            }) => hir::WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                bound_generic_params: self
                    .lower_generic_params(bound_generic_params, hir::GenericParamSource::Binder),
                bounded_ty: self
                    .lower_ty(bounded_ty, ImplTraitContext::Disallowed(ImplTraitPosition::Bound)),
                bounds: self.lower_param_bounds(
                    bounds,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                ),
                origin: PredicateOrigin::WhereClause,
            }),
            WherePredicateKind::RegionPredicate(WhereRegionPredicate { lifetime, bounds }) => {
                hir::WherePredicateKind::RegionPredicate(hir::WhereRegionPredicate {
                    lifetime: self.lower_lifetime(lifetime),
                    bounds: self.lower_param_bounds(
                        bounds,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                    ),
                    in_where_clause: true,
                })
            }
            WherePredicateKind::EqPredicate(WhereEqPredicate { lhs_ty, rhs_ty }) => {
                hir::WherePredicateKind::EqPredicate(hir::WhereEqPredicate {
                    lhs_ty: self
                        .lower_ty(lhs_ty, ImplTraitContext::Disallowed(ImplTraitPosition::Bound)),
                    rhs_ty: self
                        .lower_ty(rhs_ty, ImplTraitContext::Disallowed(ImplTraitPosition::Bound)),
                })
            }
        });
        hir::WherePredicate { hir_id, span, kind }
    }
}
