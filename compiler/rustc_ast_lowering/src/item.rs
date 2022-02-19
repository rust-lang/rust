use super::{AnonymousLifetimeMode, LoweringContext, ParamMode};
use super::{ImplTraitContext, ImplTraitPosition};
use crate::{Arena, FnDeclKind};

use rustc_ast::ptr::P;
use rustc_ast::visit::{self, AssocCtxt, FnCtxt, FnKind, Visitor};
use rustc_ast::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_index::vec::Idx;
use rustc_span::source_map::{respan, DesugaringKind};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use rustc_target::spec::abi;
use smallvec::{smallvec, SmallVec};
use tracing::debug;

use std::iter;
use std::mem;

pub(super) struct ItemLowerer<'a, 'lowering, 'hir> {
    pub(super) lctx: &'a mut LoweringContext<'lowering, 'hir>,
}

impl ItemLowerer<'_, '_, '_> {
    fn with_trait_impl_ref<T>(
        &mut self,
        impl_ref: &Option<TraitRef>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old = self.lctx.is_in_trait_impl;
        self.lctx.is_in_trait_impl = impl_ref.is_some();
        let ret = f(self);
        self.lctx.is_in_trait_impl = old;
        ret
    }
}

impl<'a> Visitor<'a> for ItemLowerer<'a, '_, '_> {
    fn visit_attribute(&mut self, _: &'a Attribute) {
        // We do not want to lower expressions that appear in attributes,
        // as they are not accessible to the rest of the HIR.
    }

    fn visit_item(&mut self, item: &'a Item) {
        let hir_id = self.lctx.with_hir_id_owner(item.id, |lctx| {
            let node = lctx.without_in_scope_lifetime_defs(|lctx| lctx.lower_item(item));
            hir::OwnerNode::Item(node)
        });

        self.lctx.with_parent_item_lifetime_defs(hir_id, |this| {
            let this = &mut ItemLowerer { lctx: this };
            match item.kind {
                ItemKind::Impl(box Impl { ref of_trait, .. }) => {
                    this.with_trait_impl_ref(of_trait, |this| visit::walk_item(this, item));
                }
                _ => visit::walk_item(this, item),
            }
        });
    }

    fn visit_fn(&mut self, fk: FnKind<'a>, sp: Span, _: NodeId) {
        match fk {
            FnKind::Fn(FnCtxt::Foreign, _, sig, _, _) => {
                self.visit_fn_header(&sig.header);
                visit::walk_fn_decl(self, &sig.decl);
                // Don't visit the foreign function body even if it has one, since lowering the
                // body would have no meaning and will have already been caught as a parse error.
            }
            _ => visit::walk_fn(self, fk, sp),
        }
    }

    fn visit_assoc_item(&mut self, item: &'a AssocItem, ctxt: AssocCtxt) {
        self.lctx.with_hir_id_owner(item.id, |lctx| match ctxt {
            AssocCtxt::Trait => hir::OwnerNode::TraitItem(lctx.lower_trait_item(item)),
            AssocCtxt::Impl => hir::OwnerNode::ImplItem(lctx.lower_impl_item(item)),
        });

        visit::walk_assoc_item(self, item, ctxt);
    }

    fn visit_foreign_item(&mut self, item: &'a ForeignItem) {
        self.lctx.with_hir_id_owner(item.id, |lctx| {
            hir::OwnerNode::ForeignItem(lctx.lower_foreign_item(item))
        });

        visit::walk_foreign_item(self, item);
    }
}

impl<'hir> LoweringContext<'_, 'hir> {
    // Same as the method above, but accepts `hir::GenericParam`s
    // instead of `ast::GenericParam`s.
    // This should only be used with generics that have already had their
    // in-band lifetimes added. In practice, this means that this function is
    // only used when lowering a child item of a trait or impl.
    fn with_parent_item_lifetime_defs<T>(
        &mut self,
        parent_hir_id: LocalDefId,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old_len = self.in_scope_lifetimes.len();

        let parent_generics = match self.owners[parent_hir_id].unwrap().node().expect_item().kind {
            hir::ItemKind::Impl(hir::Impl { ref generics, .. })
            | hir::ItemKind::Trait(_, _, ref generics, ..) => generics.params,
            _ => &[],
        };
        let lt_def_names = parent_generics.iter().filter_map(|param| match param.kind {
            hir::GenericParamKind::Lifetime { .. } => Some(param.name.normalize_to_macros_2_0()),
            _ => None,
        });
        self.in_scope_lifetimes.extend(lt_def_names);

        let res = f(self);

        self.in_scope_lifetimes.truncate(old_len);
        res
    }

    // Clears (and restores) the `in_scope_lifetimes` field. Used when
    // visiting nested items, which never inherit in-scope lifetimes
    // from their surrounding environment.
    fn without_in_scope_lifetime_defs<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let old_in_scope_lifetimes = mem::replace(&mut self.in_scope_lifetimes, vec![]);

        // this vector is only used when walking over impl headers,
        // input types, and the like, and should not be non-empty in
        // between items
        assert!(self.lifetimes_to_define.is_empty());

        let res = f(self);

        assert!(self.in_scope_lifetimes.is_empty());
        self.in_scope_lifetimes = old_in_scope_lifetimes;

        res
    }

    pub(super) fn lower_mod(&mut self, items: &[P<Item>], inner: Span) -> hir::Mod<'hir> {
        hir::Mod {
            inner: self.lower_span(inner),
            item_ids: self.arena.alloc_from_iter(items.iter().flat_map(|x| self.lower_item_ref(x))),
        }
    }

    pub(super) fn lower_item_ref(&mut self, i: &Item) -> SmallVec<[hir::ItemId; 1]> {
        let mut node_ids = smallvec![hir::ItemId { def_id: self.resolver.local_def_id(i.id) }];
        if let ItemKind::Use(ref use_tree) = &i.kind {
            self.lower_item_id_use_tree(use_tree, i.id, &mut node_ids);
        }
        node_ids
    }

    fn lower_item_id_use_tree(
        &mut self,
        tree: &UseTree,
        base_id: NodeId,
        vec: &mut SmallVec<[hir::ItemId; 1]>,
    ) {
        match tree.kind {
            UseTreeKind::Nested(ref nested_vec) => {
                for &(ref nested, id) in nested_vec {
                    vec.push(hir::ItemId { def_id: self.resolver.local_def_id(id) });
                    self.lower_item_id_use_tree(nested, id, vec);
                }
            }
            UseTreeKind::Glob => {}
            UseTreeKind::Simple(_, id1, id2) => {
                for (_, &id) in
                    iter::zip(self.expect_full_res_from_use(base_id).skip(1), &[id1, id2])
                {
                    vec.push(hir::ItemId { def_id: self.resolver.local_def_id(id) });
                }
            }
        }
    }

    fn lower_item(&mut self, i: &Item) -> &'hir hir::Item<'hir> {
        let mut ident = i.ident;
        let mut vis = self.lower_visibility(&i.vis);
        let hir_id = self.lower_node_id(i.id);
        let attrs = self.lower_attrs(hir_id, &i.attrs);
        let kind = self.lower_item_kind(i.span, i.id, hir_id, &mut ident, attrs, &mut vis, &i.kind);
        let item = hir::Item {
            def_id: hir_id.expect_owner(),
            ident: self.lower_ident(ident),
            kind,
            vis,
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
        attrs: Option<&'hir [Attribute]>,
        vis: &mut hir::Visibility<'hir>,
        i: &ItemKind,
    ) -> hir::ItemKind<'hir> {
        match *i {
            ItemKind::ExternCrate(orig_name) => hir::ItemKind::ExternCrate(orig_name),
            ItemKind::Use(ref use_tree) => {
                // Start with an empty prefix.
                let prefix = Path { segments: vec![], span: use_tree.span, tokens: None };

                self.lower_use_tree(use_tree, &prefix, id, vis, ident, attrs)
            }
            ItemKind::Static(ref t, m, ref e) => {
                let (ty, body_id) = self.lower_const_item(t, span, e.as_deref());
                hir::ItemKind::Static(ty, m, body_id)
            }
            ItemKind::Const(_, ref t, ref e) => {
                let (ty, body_id) = self.lower_const_item(t, span, e.as_deref());
                hir::ItemKind::Const(ty, body_id)
            }
            ItemKind::Fn(box Fn {
                sig: FnSig { ref decl, header, span: fn_sig_span },
                ref generics,
                ref body,
                ..
            }) => {
                let fn_def_id = self.resolver.local_def_id(id);
                self.with_new_scopes(|this| {
                    this.current_item = Some(ident.span);

                    // Note: we don't need to change the return type from `T` to
                    // `impl Future<Output = T>` here because lower_body
                    // only cares about the input argument patterns in the function
                    // declaration (decl), not the return types.
                    let asyncness = header.asyncness;
                    let body_id =
                        this.lower_maybe_async_body(span, &decl, asyncness, body.as_deref());

                    let (generics, decl) = this.add_in_band_defs(
                        generics,
                        fn_def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, idty| {
                            let ret_id = asyncness.opt_return_id();
                            this.lower_fn_decl(
                                &decl,
                                Some((fn_def_id, idty)),
                                FnDeclKind::Fn,
                                ret_id,
                            )
                        },
                    );
                    let sig = hir::FnSig {
                        decl,
                        header: this.lower_fn_header(header),
                        span: this.lower_span(fn_sig_span),
                    };
                    hir::ItemKind::Fn(sig, generics, body_id)
                })
            }
            ItemKind::Mod(_, ref mod_kind) => match mod_kind {
                ModKind::Loaded(items, _, inner_span) => {
                    hir::ItemKind::Mod(self.lower_mod(items, *inner_span))
                }
                ModKind::Unloaded => panic!("`mod` items should have been loaded by now"),
            },
            ItemKind::ForeignMod(ref fm) => hir::ItemKind::ForeignMod {
                abi: fm.abi.map_or(abi::Abi::FALLBACK, |abi| self.lower_abi(abi)),
                items: self
                    .arena
                    .alloc_from_iter(fm.items.iter().map(|x| self.lower_foreign_item_ref(x))),
            },
            ItemKind::GlobalAsm(ref asm) => {
                hir::ItemKind::GlobalAsm(self.lower_inline_asm(span, asm))
            }
            ItemKind::TyAlias(box TyAlias { ref generics, ty: Some(ref ty), .. }) => {
                // We lower
                //
                // type Foo = impl Trait
                //
                // to
                //
                // type Foo = Foo1
                // opaque type Foo1: Trait
                let ty = self.lower_ty(
                    ty,
                    ImplTraitContext::TypeAliasesOpaqueTy {
                        capturable_lifetimes: &mut FxHashSet::default(),
                    },
                );
                let generics = self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                );
                hir::ItemKind::TyAlias(ty, generics)
            }
            ItemKind::TyAlias(box TyAlias { ref generics, ty: None, .. }) => {
                let ty = self.arena.alloc(self.ty(span, hir::TyKind::Err));
                let generics = self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                );
                hir::ItemKind::TyAlias(ty, generics)
            }
            ItemKind::Enum(ref enum_definition, ref generics) => hir::ItemKind::Enum(
                hir::EnumDef {
                    variants: self.arena.alloc_from_iter(
                        enum_definition.variants.iter().map(|x| self.lower_variant(x)),
                    ),
                },
                self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                ),
            ),
            ItemKind::Struct(ref struct_def, ref generics) => {
                let struct_def = self.lower_variant_data(hir_id, struct_def);
                hir::ItemKind::Struct(
                    struct_def,
                    self.lower_generics(
                        generics,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    ),
                )
            }
            ItemKind::Union(ref vdata, ref generics) => {
                let vdata = self.lower_variant_data(hir_id, vdata);
                hir::ItemKind::Union(
                    vdata,
                    self.lower_generics(
                        generics,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    ),
                )
            }
            ItemKind::Impl(box Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                generics: ref ast_generics,
                of_trait: ref trait_ref,
                self_ty: ref ty,
                items: ref impl_items,
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
                let lowered_trait_def_id = self.lower_node_id(id).expect_owner();
                let (generics, (trait_ref, lowered_ty)) = self.add_in_band_defs(
                    ast_generics,
                    lowered_trait_def_id,
                    AnonymousLifetimeMode::CreateParameter,
                    |this, _| {
                        let trait_ref = trait_ref.as_ref().map(|trait_ref| {
                            this.lower_trait_ref(
                                trait_ref,
                                ImplTraitContext::Disallowed(ImplTraitPosition::Trait),
                            )
                        });

                        let lowered_ty = this
                            .lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type));

                        (trait_ref, lowered_ty)
                    },
                );

                let new_impl_items =
                    self.with_in_scope_lifetime_defs(&ast_generics.params, |this| {
                        this.arena.alloc_from_iter(
                            impl_items.iter().map(|item| this.lower_impl_item_ref(item)),
                        )
                    });

                // `defaultness.has_value()` is never called for an `impl`, always `true` in order
                // to not cause an assertion failure inside the `lower_defaultness` function.
                let has_val = true;
                let (defaultness, defaultness_span) = self.lower_defaultness(defaultness, has_val);
                let polarity = match polarity {
                    ImplPolarity::Positive => ImplPolarity::Positive,
                    ImplPolarity::Negative(s) => ImplPolarity::Negative(self.lower_span(s)),
                };
                hir::ItemKind::Impl(hir::Impl {
                    unsafety: self.lower_unsafety(unsafety),
                    polarity,
                    defaultness,
                    defaultness_span,
                    constness: self.lower_constness(constness),
                    generics,
                    of_trait: trait_ref,
                    self_ty: lowered_ty,
                    items: new_impl_items,
                })
            }
            ItemKind::Trait(box Trait {
                is_auto,
                unsafety,
                ref generics,
                ref bounds,
                ref items,
            }) => {
                let bounds = self.lower_param_bounds(
                    bounds,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                );
                let items = self
                    .arena
                    .alloc_from_iter(items.iter().map(|item| self.lower_trait_item_ref(item)));
                hir::ItemKind::Trait(
                    is_auto,
                    self.lower_unsafety(unsafety),
                    self.lower_generics(
                        generics,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    ),
                    bounds,
                    items,
                )
            }
            ItemKind::TraitAlias(ref generics, ref bounds) => hir::ItemKind::TraitAlias(
                self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                ),
                self.lower_param_bounds(
                    bounds,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                ),
            ),
            ItemKind::MacroDef(MacroDef { ref body, macro_rules }) => {
                let body = P(self.lower_mac_args(body));

                hir::ItemKind::Macro(ast::MacroDef { body, macro_rules })
            }
            ItemKind::MacCall(..) => {
                panic!("`TyMac` should have been expanded by now")
            }
        }
    }

    fn lower_const_item(
        &mut self,
        ty: &Ty,
        span: Span,
        body: Option<&Expr>,
    ) -> (&'hir hir::Ty<'hir>, hir::BodyId) {
        let ty = self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type));
        (ty, self.lower_const_body(span, body))
    }

    fn lower_use_tree(
        &mut self,
        tree: &UseTree,
        prefix: &Path,
        id: NodeId,
        vis: &mut hir::Visibility<'hir>,
        ident: &mut Ident,
        attrs: Option<&'hir [Attribute]>,
    ) -> hir::ItemKind<'hir> {
        debug!("lower_use_tree(tree={:?})", tree);
        debug!("lower_use_tree: vis = {:?}", vis);

        let path = &tree.prefix;
        let segments = prefix.segments.iter().chain(path.segments.iter()).cloned().collect();

        match tree.kind {
            UseTreeKind::Simple(rename, id1, id2) => {
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

                let mut resolutions = self.expect_full_res_from_use(id).fuse();
                // We want to return *something* from this function, so hold onto the first item
                // for later.
                let ret_res = self.lower_res(resolutions.next().unwrap_or(Res::Err));

                // Here, we are looping over namespaces, if they exist for the definition
                // being imported. We only handle type and value namespaces because we
                // won't be dealing with macros in the rest of the compiler.
                // Essentially a single `use` which imports two names is desugared into
                // two imports.
                for new_node_id in [id1, id2] {
                    let new_id = self.resolver.local_def_id(new_node_id);
                    let Some(res) = resolutions.next() else {
                        // Associate an HirId to both ids even if there is no resolution.
                        let _old = self
                            .node_id_to_hir_id
                            .insert(new_node_id, hir::HirId::make_owner(new_id));
                        debug_assert!(_old.is_none());
                        self.owners.ensure_contains_elem(new_id, || hir::MaybeOwner::Phantom);
                        let _old = std::mem::replace(
                            &mut self.owners[new_id],
                            hir::MaybeOwner::NonOwner(hir::HirId::make_owner(new_id)),
                        );
                        debug_assert!(matches!(_old, hir::MaybeOwner::Phantom));
                        continue;
                    };
                    let ident = *ident;
                    let mut path = path.clone();
                    for seg in &mut path.segments {
                        seg.id = self.resolver.next_node_id();
                    }
                    let span = path.span;

                    self.with_hir_id_owner(new_node_id, |this| {
                        let res = this.lower_res(res);
                        let path = this.lower_path_extra(res, &path, ParamMode::Explicit);
                        let kind = hir::ItemKind::Use(path, hir::UseKind::Single);
                        let vis = this.rebuild_vis(&vis);
                        if let Some(attrs) = attrs {
                            this.attrs.insert(hir::ItemLocalId::new(0), attrs);
                        }

                        let item = hir::Item {
                            def_id: new_id,
                            ident: this.lower_ident(ident),
                            kind,
                            vis,
                            span: this.lower_span(span),
                        };
                        hir::OwnerNode::Item(this.arena.alloc(item))
                    });
                }

                let path = self.lower_path_extra(ret_res, &path, ParamMode::Explicit);
                hir::ItemKind::Use(path, hir::UseKind::Single)
            }
            UseTreeKind::Glob => {
                let path = self.lower_path(
                    id,
                    &Path { segments, span: path.span, tokens: None },
                    ParamMode::Explicit,
                );
                hir::ItemKind::Use(path, hir::UseKind::Glob)
            }
            UseTreeKind::Nested(ref trees) => {
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
                // like `use foo::{a::{b, c}}` and so forth).  They
                // wind up being directly added to
                // `self.items`. However, the structure of this
                // function also requires us to return one item, and
                // for that we return the `{}` import (called the
                // `ListStem`).

                let prefix = Path { segments, span: prefix.span.to(path.span), tokens: None };

                // Add all the nested `PathListItem`s to the HIR.
                for &(ref use_tree, id) in trees {
                    let new_hir_id = self.resolver.local_def_id(id);

                    let mut prefix = prefix.clone();

                    // Give the segments new node-ids since they are being cloned.
                    for seg in &mut prefix.segments {
                        seg.id = self.resolver.next_node_id();
                    }

                    // Each `use` import is an item and thus are owners of the
                    // names in the path. Up to this point the nested import is
                    // the current owner, since we want each desugared import to
                    // own its own names, we have to adjust the owner before
                    // lowering the rest of the import.
                    self.with_hir_id_owner(id, |this| {
                        let mut vis = this.rebuild_vis(&vis);
                        let mut ident = *ident;

                        let kind =
                            this.lower_use_tree(use_tree, &prefix, id, &mut vis, &mut ident, attrs);
                        if let Some(attrs) = attrs {
                            this.attrs.insert(hir::ItemLocalId::new(0), attrs);
                        }

                        let item = hir::Item {
                            def_id: new_hir_id,
                            ident: this.lower_ident(ident),
                            kind,
                            vis,
                            span: this.lower_span(use_tree.span),
                        };
                        hir::OwnerNode::Item(this.arena.alloc(item))
                    });
                }

                // Subtle and a bit hacky: we lower the privacy level
                // of the list stem to "private" most of the time, but
                // not for "restricted" paths. The key thing is that
                // we don't want it to stay as `pub` (with no caveats)
                // because that affects rustdoc and also the lints
                // about `pub` items. But we can't *always* make it
                // private -- particularly not for restricted paths --
                // because it contains node-ids that would then be
                // unused, failing the check that HirIds are "densely
                // assigned".
                match vis.node {
                    hir::VisibilityKind::Public
                    | hir::VisibilityKind::Crate(_)
                    | hir::VisibilityKind::Inherited => {
                        *vis = respan(
                            self.lower_span(prefix.span.shrink_to_lo()),
                            hir::VisibilityKind::Inherited,
                        );
                    }
                    hir::VisibilityKind::Restricted { .. } => {
                        // Do nothing here, as described in the comment on the match.
                    }
                }

                let res = self.expect_full_res_from_use(id).next().unwrap_or(Res::Err);
                let res = self.lower_res(res);
                let path = self.lower_path_extra(res, &prefix, ParamMode::Explicit);
                hir::ItemKind::Use(path, hir::UseKind::ListStem)
            }
        }
    }

    /// Paths like the visibility path in `pub(super) use foo::{bar, baz}` are repeated
    /// many times in the HIR tree; for each occurrence, we need to assign distinct
    /// `NodeId`s. (See, e.g., #56128.)
    fn rebuild_use_path(&mut self, path: &hir::Path<'hir>) -> &'hir hir::Path<'hir> {
        debug!("rebuild_use_path(path = {:?})", path);
        let segments =
            self.arena.alloc_from_iter(path.segments.iter().map(|seg| hir::PathSegment {
                ident: seg.ident,
                hir_id: seg.hir_id.map(|_| self.next_id()),
                res: seg.res,
                args: None,
                infer_args: seg.infer_args,
            }));
        self.arena.alloc(hir::Path { span: path.span, res: path.res, segments })
    }

    fn rebuild_vis(&mut self, vis: &hir::Visibility<'hir>) -> hir::Visibility<'hir> {
        let vis_kind = match vis.node {
            hir::VisibilityKind::Public => hir::VisibilityKind::Public,
            hir::VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            hir::VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
            hir::VisibilityKind::Restricted { ref path, hir_id: _ } => {
                hir::VisibilityKind::Restricted {
                    path: self.rebuild_use_path(path),
                    hir_id: self.next_id(),
                }
            }
        };
        respan(self.lower_span(vis.span), vis_kind)
    }

    fn lower_foreign_item(&mut self, i: &ForeignItem) -> &'hir hir::ForeignItem<'hir> {
        let hir_id = self.lower_node_id(i.id);
        let def_id = hir_id.expect_owner();
        self.lower_attrs(hir_id, &i.attrs);
        let item = hir::ForeignItem {
            def_id,
            ident: self.lower_ident(i.ident),
            kind: match i.kind {
                ForeignItemKind::Fn(box Fn { ref sig, ref generics, .. }) => {
                    let fdec = &sig.decl;
                    let (generics, (fn_dec, fn_args)) = self.add_in_band_defs(
                        generics,
                        def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, _| {
                            (
                                // Disallow `impl Trait` in foreign items.
                                this.lower_fn_decl(fdec, None, FnDeclKind::ExternFn, None),
                                this.lower_fn_params_to_names(fdec),
                            )
                        },
                    );

                    hir::ForeignItemKind::Fn(fn_dec, fn_args, generics)
                }
                ForeignItemKind::Static(ref t, m, _) => {
                    let ty =
                        self.lower_ty(t, ImplTraitContext::Disallowed(ImplTraitPosition::Type));
                    hir::ForeignItemKind::Static(ty, m)
                }
                ForeignItemKind::TyAlias(..) => hir::ForeignItemKind::Type,
                ForeignItemKind::MacCall(_) => panic!("macro shouldn't exist here"),
            },
            vis: self.lower_visibility(&i.vis),
            span: self.lower_span(i.span),
        };
        self.arena.alloc(item)
    }

    fn lower_foreign_item_ref(&mut self, i: &ForeignItem) -> hir::ForeignItemRef {
        hir::ForeignItemRef {
            id: hir::ForeignItemId { def_id: self.resolver.local_def_id(i.id) },
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
        }
    }

    fn lower_variant(&mut self, v: &Variant) -> hir::Variant<'hir> {
        let id = self.lower_node_id(v.id);
        self.lower_attrs(id, &v.attrs);
        hir::Variant {
            id,
            data: self.lower_variant_data(id, &v.data),
            disr_expr: v.disr_expr.as_ref().map(|e| self.lower_anon_const(e)),
            ident: self.lower_ident(v.ident),
            span: self.lower_span(v.span),
        }
    }

    fn lower_variant_data(
        &mut self,
        parent_id: hir::HirId,
        vdata: &VariantData,
    ) -> hir::VariantData<'hir> {
        match *vdata {
            VariantData::Struct(ref fields, recovered) => hir::VariantData::Struct(
                self.arena
                    .alloc_from_iter(fields.iter().enumerate().map(|f| self.lower_field_def(f))),
                recovered,
            ),
            VariantData::Tuple(ref fields, id) => {
                let ctor_id = self.lower_node_id(id);
                self.alias_attrs(ctor_id, parent_id);
                hir::VariantData::Tuple(
                    self.arena.alloc_from_iter(
                        fields.iter().enumerate().map(|f| self.lower_field_def(f)),
                    ),
                    ctor_id,
                )
            }
            VariantData::Unit(id) => {
                let ctor_id = self.lower_node_id(id);
                self.alias_attrs(ctor_id, parent_id);
                hir::VariantData::Unit(ctor_id)
            }
        }
    }

    fn lower_field_def(&mut self, (index, f): (usize, &FieldDef)) -> hir::FieldDef<'hir> {
        let ty = if let TyKind::Path(ref qself, ref path) = f.ty.kind {
            let t = self.lower_path_ty(
                &f.ty,
                qself,
                path,
                ParamMode::ExplicitNamed, // no `'_` in declarations (Issue #61124)
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
            );
            self.arena.alloc(t)
        } else {
            self.lower_ty(&f.ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type))
        };
        let hir_id = self.lower_node_id(f.id);
        self.lower_attrs(hir_id, &f.attrs);
        hir::FieldDef {
            span: self.lower_span(f.span),
            hir_id,
            ident: match f.ident {
                Some(ident) => self.lower_ident(ident),
                // FIXME(jseyfried): positional field hygiene.
                None => Ident::new(sym::integer(index), self.lower_span(f.span)),
            },
            vis: self.lower_visibility(&f.vis),
            ty,
        }
    }

    fn lower_trait_item(&mut self, i: &AssocItem) -> &'hir hir::TraitItem<'hir> {
        let hir_id = self.lower_node_id(i.id);
        let trait_item_def_id = hir_id.expect_owner();

        let (generics, kind) = match i.kind {
            AssocItemKind::Const(_, ref ty, ref default) => {
                let ty = self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type));
                let body = default.as_ref().map(|x| self.lower_const_body(i.span, Some(x)));
                (hir::Generics::empty(), hir::TraitItemKind::Const(ty, body))
            }
            AssocItemKind::Fn(box Fn { ref sig, ref generics, body: None, .. }) => {
                let names = self.lower_fn_params_to_names(&sig.decl);
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    trait_item_def_id,
                    FnDeclKind::Trait,
                    None,
                );
                (generics, hir::TraitItemKind::Fn(sig, hir::TraitFn::Required(names)))
            }
            AssocItemKind::Fn(box Fn { ref sig, ref generics, body: Some(ref body), .. }) => {
                let asyncness = sig.header.asyncness;
                let body_id =
                    self.lower_maybe_async_body(i.span, &sig.decl, asyncness, Some(&body));
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    trait_item_def_id,
                    FnDeclKind::Trait,
                    asyncness.opt_return_id(),
                );
                (generics, hir::TraitItemKind::Fn(sig, hir::TraitFn::Provided(body_id)))
            }
            AssocItemKind::TyAlias(box TyAlias { ref generics, ref bounds, ref ty, .. }) => {
                let ty = ty.as_ref().map(|x| {
                    self.lower_ty(x, ImplTraitContext::Disallowed(ImplTraitPosition::Type))
                });
                let generics = self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                );
                let kind = hir::TraitItemKind::Type(
                    self.lower_param_bounds(
                        bounds,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                    ),
                    ty,
                );

                (generics, kind)
            }
            AssocItemKind::MacCall(..) => panic!("macro item shouldn't exist at this point"),
        };

        self.lower_attrs(hir_id, &i.attrs);
        let item = hir::TraitItem {
            def_id: trait_item_def_id,
            ident: self.lower_ident(i.ident),
            generics,
            kind,
            span: self.lower_span(i.span),
        };
        self.arena.alloc(item)
    }

    fn lower_trait_item_ref(&mut self, i: &AssocItem) -> hir::TraitItemRef {
        let (kind, has_default) = match &i.kind {
            AssocItemKind::Const(_, _, default) => (hir::AssocItemKind::Const, default.is_some()),
            AssocItemKind::TyAlias(box TyAlias { ty, .. }) => {
                (hir::AssocItemKind::Type, ty.is_some())
            }
            AssocItemKind::Fn(box Fn { sig, body, .. }) => {
                (hir::AssocItemKind::Fn { has_self: sig.decl.has_self() }, body.is_some())
            }
            AssocItemKind::MacCall(..) => unimplemented!(),
        };
        let id = hir::TraitItemId { def_id: self.resolver.local_def_id(i.id) };
        let defaultness = hir::Defaultness::Default { has_value: has_default };
        hir::TraitItemRef {
            id,
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
            defaultness,
            kind,
        }
    }

    /// Construct `ExprKind::Err` for the given `span`.
    crate fn expr_err(&mut self, span: Span) -> hir::Expr<'hir> {
        self.expr(span, hir::ExprKind::Err, AttrVec::new())
    }

    fn lower_impl_item(&mut self, i: &AssocItem) -> &'hir hir::ImplItem<'hir> {
        let impl_item_def_id = self.resolver.local_def_id(i.id);

        let (generics, kind) = match &i.kind {
            AssocItemKind::Const(_, ty, expr) => {
                let ty = self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type));
                (
                    hir::Generics::empty(),
                    hir::ImplItemKind::Const(ty, self.lower_const_body(i.span, expr.as_deref())),
                )
            }
            AssocItemKind::Fn(box Fn { sig, generics, body, .. }) => {
                self.current_item = Some(i.span);
                let asyncness = sig.header.asyncness;
                let body_id =
                    self.lower_maybe_async_body(i.span, &sig.decl, asyncness, body.as_deref());
                let (generics, sig) = self.lower_method_sig(
                    generics,
                    sig,
                    impl_item_def_id,
                    if self.is_in_trait_impl { FnDeclKind::Impl } else { FnDeclKind::Inherent },
                    asyncness.opt_return_id(),
                );

                (generics, hir::ImplItemKind::Fn(sig, body_id))
            }
            AssocItemKind::TyAlias(box TyAlias { generics, ty, .. }) => {
                let generics = self.lower_generics(
                    generics,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                );
                let kind = match ty {
                    None => {
                        let ty = self.arena.alloc(self.ty(i.span, hir::TyKind::Err));
                        hir::ImplItemKind::TyAlias(ty)
                    }
                    Some(ty) => {
                        let ty = self.lower_ty(
                            ty,
                            ImplTraitContext::TypeAliasesOpaqueTy {
                                capturable_lifetimes: &mut FxHashSet::default(),
                            },
                        );
                        hir::ImplItemKind::TyAlias(ty)
                    }
                };
                (generics, kind)
            }
            AssocItemKind::MacCall(..) => panic!("`TyMac` should have been expanded by now"),
        };

        let hir_id = self.lower_node_id(i.id);
        self.lower_attrs(hir_id, &i.attrs);
        let item = hir::ImplItem {
            def_id: hir_id.expect_owner(),
            ident: self.lower_ident(i.ident),
            generics,
            vis: self.lower_visibility(&i.vis),
            kind,
            span: self.lower_span(i.span),
        };
        self.arena.alloc(item)
    }

    fn lower_impl_item_ref(&mut self, i: &AssocItem) -> hir::ImplItemRef {
        // Since `default impl` is not yet implemented, this is always true in impls.
        let has_value = true;
        let (defaultness, _) = self.lower_defaultness(i.kind.defaultness(), has_value);
        hir::ImplItemRef {
            id: hir::ImplItemId { def_id: self.resolver.local_def_id(i.id) },
            ident: self.lower_ident(i.ident),
            span: self.lower_span(i.span),
            defaultness,
            kind: match &i.kind {
                AssocItemKind::Const(..) => hir::AssocItemKind::Const,
                AssocItemKind::TyAlias(..) => hir::AssocItemKind::Type,
                AssocItemKind::Fn(box Fn { sig, .. }) => {
                    hir::AssocItemKind::Fn { has_self: sig.decl.has_self() }
                }
                AssocItemKind::MacCall(..) => unimplemented!(),
            },
            trait_item_def_id: self.resolver.get_partial_res(i.id).map(|r| r.base_res().def_id()),
        }
    }

    /// If an `explicit_owner` is given, this method allocates the `HirId` in
    /// the address space of that item instead of the item currently being
    /// lowered. This can happen during `lower_impl_item_ref()` where we need to
    /// lower a `Visibility` value although we haven't lowered the owning
    /// `ImplItem` in question yet.
    fn lower_visibility(&mut self, v: &Visibility) -> hir::Visibility<'hir> {
        let node = match v.kind {
            VisibilityKind::Public => hir::VisibilityKind::Public,
            VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            VisibilityKind::Restricted { ref path, id } => {
                debug!("lower_visibility: restricted path id = {:?}", id);
                let lowered_id = self.lower_node_id(id);
                hir::VisibilityKind::Restricted {
                    path: self.lower_path(id, path, ParamMode::Explicit),
                    hir_id: lowered_id,
                }
            }
            VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
        };
        respan(self.lower_span(v.span), node)
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
        let body = hir::Body { generator_kind: self.generator_kind, params, value };
        let id = body.id();
        debug_assert_eq!(id.hir_id.owner, self.current_hir_id_owner);
        self.bodies.push((id.hir_id.local_id, self.arena.alloc(body)));
        id
    }

    pub(super) fn lower_body(
        &mut self,
        f: impl FnOnce(&mut Self) -> (&'hir [hir::Param<'hir>], hir::Expr<'hir>),
    ) -> hir::BodyId {
        let prev_gen_kind = self.generator_kind.take();
        let task_context = self.task_context.take();
        let (parameters, result) = f(self);
        let body_id = self.record_body(parameters, result);
        self.task_context = task_context;
        self.generator_kind = prev_gen_kind;
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
            None => self.expr_err(span),
        }
    }

    pub(super) fn lower_const_body(&mut self, span: Span, expr: Option<&Expr>) -> hir::BodyId {
        self.lower_body(|this| {
            (
                &[],
                match expr {
                    Some(expr) => this.lower_expr_mut(expr),
                    None => this.expr_err(span),
                },
            )
        })
    }

    fn lower_maybe_async_body(
        &mut self,
        span: Span,
        decl: &FnDecl,
        asyncness: Async,
        body: Option<&Block>,
    ) -> hir::BodyId {
        let closure_id = match asyncness {
            Async::Yes { closure_id, .. } => closure_id,
            Async::No => return self.lower_fn_body_block(span, decl, body),
        };

        self.lower_body(|this| {
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
                let parameter = this.lower_param(parameter);
                let span = parameter.pat.span;

                // Check if this is a binding pattern, if so, we can optimize and avoid adding a
                // `let <pat> = __argN;` statement. In this case, we do not rename the parameter.
                let (ident, is_simple_parameter) = match parameter.pat.kind {
                    hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated | hir::BindingAnnotation::Mutable,
                        _,
                        ident,
                        _,
                    ) => (ident, true),
                    // For `ref mut` or wildcard arguments, we can't reuse the binding, but
                    // we can keep the same name for the parameter.
                    // This lets rustdoc render it correctly in documentation.
                    hir::PatKind::Binding(_, _, ident, _) => (ident, false),
                    hir::PatKind::Wild => {
                        (Ident::with_dummy_span(rustc_span::symbol::kw::Underscore), false)
                    }
                    _ => {
                        // Replace the ident for bindings that aren't simple.
                        let name = format!("__arg{}", index);
                        let ident = Ident::from_str(&name);

                        (ident, false)
                    }
                };

                let desugared_span = this.mark_span_with_reason(DesugaringKind::Async, span, None);

                // Construct a parameter representing `__argN: <ty>` to replace the parameter of the
                // async function.
                //
                // If this is the simple case, this parameter will end up being the same as the
                // original parameter, but with a different pattern id.
                let stmt_attrs = this.attrs.get(&parameter.hir_id.local_id).copied();
                let (new_parameter_pat, new_parameter_id) = this.pat_ident(desugared_span, ident);
                let new_parameter = hir::Param {
                    hir_id: parameter.hir_id,
                    pat: new_parameter_pat,
                    ty_span: this.lower_span(parameter.ty_span),
                    span: this.lower_span(parameter.span),
                };

                if is_simple_parameter {
                    // If this is the simple case, then we only insert one statement that is
                    // `let <pat> = <pat>;`. We re-use the original argument's pattern so that
                    // `HirId`s are densely assigned.
                    let expr = this.expr_ident(desugared_span, ident, new_parameter_id);
                    let stmt = this.stmt_let_pat(
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
                    let (move_pat, move_id) = this.pat_ident_binding_mode(
                        desugared_span,
                        ident,
                        hir::BindingAnnotation::Mutable,
                    );
                    let move_expr = this.expr_ident(desugared_span, ident, new_parameter_id);
                    let move_stmt = this.stmt_let_pat(
                        None,
                        desugared_span,
                        Some(move_expr),
                        move_pat,
                        hir::LocalSource::AsyncFn,
                    );

                    // Construct the `let <pat> = __argN;` statement. We re-use the original
                    // parameter's pattern so that `HirId`s are densely assigned.
                    let pattern_expr = this.expr_ident(desugared_span, ident, move_id);
                    let pattern_stmt = this.stmt_let_pat(
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

            let body_span = body.map_or(span, |b| b.span);
            let async_expr = this.make_async_expr(
                CaptureBy::Value,
                closure_id,
                None,
                body_span,
                hir::AsyncGeneratorKind::Fn,
                |this| {
                    // Create a block from the user's function body:
                    let user_body = this.lower_block_expr_opt(body_span, body);

                    // Transform into `drop-temps { <user-body> }`, an expression:
                    let desugared_span =
                        this.mark_span_with_reason(DesugaringKind::Async, user_body.span, None);
                    let user_body = this.expr_drop_temps(
                        desugared_span,
                        this.arena.alloc(user_body),
                        AttrVec::new(),
                    );

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

                    this.expr_block(body, AttrVec::new())
                },
            );

            (
                this.arena.alloc_from_iter(parameters),
                this.expr(body_span, async_expr, AttrVec::new()),
            )
        })
    }

    fn lower_method_sig(
        &mut self,
        generics: &Generics,
        sig: &FnSig,
        fn_def_id: LocalDefId,
        kind: FnDeclKind,
        is_async: Option<NodeId>,
    ) -> (hir::Generics<'hir>, hir::FnSig<'hir>) {
        let header = self.lower_fn_header(sig.header);
        let (generics, decl) = self.add_in_band_defs(
            generics,
            fn_def_id,
            AnonymousLifetimeMode::PassThrough,
            |this, idty| this.lower_fn_decl(&sig.decl, Some((fn_def_id, idty)), kind, is_async),
        );
        (generics, hir::FnSig { header, decl, span: self.lower_span(sig.span) })
    }

    fn lower_fn_header(&mut self, h: FnHeader) -> hir::FnHeader {
        hir::FnHeader {
            unsafety: self.lower_unsafety(h.unsafety),
            asyncness: self.lower_asyncness(h.asyncness),
            constness: self.lower_constness(h.constness),
            abi: self.lower_extern(h.ext),
        }
    }

    pub(super) fn lower_abi(&mut self, abi: StrLit) -> abi::Abi {
        abi::lookup(abi.symbol_unescaped.as_str()).unwrap_or_else(|| {
            self.error_on_invalid_abi(abi);
            abi::Abi::Rust
        })
    }

    pub(super) fn lower_extern(&mut self, ext: Extern) -> abi::Abi {
        match ext {
            Extern::None => abi::Abi::Rust,
            Extern::Implicit => abi::Abi::FALLBACK,
            Extern::Explicit(abi) => self.lower_abi(abi),
        }
    }

    fn error_on_invalid_abi(&self, abi: StrLit) {
        struct_span_err!(self.sess, abi.span, E0703, "invalid ABI: found `{}`", abi.symbol)
            .span_label(abi.span, "invalid ABI")
            .help(&format!("valid ABIs: {}", abi::all_names().join(", ")))
            .emit();
    }

    fn lower_asyncness(&mut self, a: Async) -> hir::IsAsync {
        match a {
            Async::Yes { .. } => hir::IsAsync::Async,
            Async::No => hir::IsAsync::NotAsync,
        }
    }

    fn lower_constness(&mut self, c: Const) -> hir::Constness {
        match c {
            Const::Yes(_) => hir::Constness::Const,
            Const::No => hir::Constness::NotConst,
        }
    }

    pub(super) fn lower_unsafety(&mut self, u: Unsafe) -> hir::Unsafety {
        match u {
            Unsafe::Yes(_) => hir::Unsafety::Unsafe,
            Unsafe::No => hir::Unsafety::Normal,
        }
    }

    pub(super) fn lower_generics_mut(
        &mut self,
        generics: &Generics,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> GenericsCtor<'hir> {
        // Error if `?Trait` bounds in where clauses don't refer directly to type paramters.
        // Note: we used to clone these bounds directly onto the type parameter (and avoid lowering
        // these into hir when we lower thee where clauses), but this makes it quite difficult to
        // keep track of the Span info. Now, `add_implicitly_sized` in `AstConv` checks both param bounds and
        // where clauses for `?Sized`.
        for pred in &generics.where_clause.predicates {
            let bound_pred = match *pred {
                WherePredicate::BoundPredicate(ref bound_pred) => bound_pred,
                _ => continue,
            };
            let compute_is_param = || {
                // Check if the where clause type is a plain type parameter.
                match self
                    .resolver
                    .get_partial_res(bound_pred.bounded_ty.id)
                    .map(|d| (d.base_res(), d.unresolved_segments()))
                {
                    Some((Res::Def(DefKind::TyParam, def_id), 0))
                        if bound_pred.bound_generic_params.is_empty() =>
                    {
                        generics
                            .params
                            .iter()
                            .any(|p| def_id == self.resolver.local_def_id(p.id).to_def_id())
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
                if !matches!(*bound, GenericBound::Trait(_, TraitBoundModifier::Maybe)) {
                    continue;
                }
                let is_param = *is_param.get_or_insert_with(compute_is_param);
                if !is_param {
                    self.diagnostic().span_err(
                        bound.span(),
                        "`?Trait` bounds are only permitted at the \
                        point where a type parameter is declared",
                    );
                }
            }
        }

        GenericsCtor {
            params: self.lower_generic_params_mut(&generics.params, itctx).collect(),
            where_clause: self.lower_where_clause(&generics.where_clause),
            span: self.lower_span(generics.span),
        }
    }

    pub(super) fn lower_generics(
        &mut self,
        generics: &Generics,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::Generics<'hir> {
        let generics_ctor = self.lower_generics_mut(generics, itctx);
        generics_ctor.into_generics(self.arena)
    }

    fn lower_where_clause(&mut self, wc: &WhereClause) -> hir::WhereClause<'hir> {
        self.with_anonymous_lifetime_mode(AnonymousLifetimeMode::ReportError, |this| {
            hir::WhereClause {
                predicates: this.arena.alloc_from_iter(
                    wc.predicates.iter().map(|predicate| this.lower_where_predicate(predicate)),
                ),
                span: this.lower_span(wc.span),
            }
        })
    }

    fn lower_where_predicate(&mut self, pred: &WherePredicate) -> hir::WherePredicate<'hir> {
        match *pred {
            WherePredicate::BoundPredicate(WhereBoundPredicate {
                ref bound_generic_params,
                ref bounded_ty,
                ref bounds,
                span,
            }) => self.with_in_scope_lifetime_defs(&bound_generic_params, |this| {
                hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                    bound_generic_params: this.lower_generic_params(
                        bound_generic_params,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Generic),
                    ),
                    bounded_ty: this.lower_ty(
                        bounded_ty,
                        ImplTraitContext::Disallowed(ImplTraitPosition::Type),
                    ),
                    bounds: this.arena.alloc_from_iter(bounds.iter().map(|bound| {
                        this.lower_param_bound(
                            bound,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                        )
                    })),
                    span: this.lower_span(span),
                })
            }),
            WherePredicate::RegionPredicate(WhereRegionPredicate {
                ref lifetime,
                ref bounds,
                span,
            }) => hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span: self.lower_span(span),
                lifetime: self.lower_lifetime(lifetime),
                bounds: self.lower_param_bounds(
                    bounds,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Bound),
                ),
            }),
            WherePredicate::EqPredicate(WhereEqPredicate { id, ref lhs_ty, ref rhs_ty, span }) => {
                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    hir_id: self.lower_node_id(id),
                    lhs_ty: self
                        .lower_ty(lhs_ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type)),
                    rhs_ty: self
                        .lower_ty(rhs_ty, ImplTraitContext::Disallowed(ImplTraitPosition::Type)),
                    span: self.lower_span(span),
                })
            }
        }
    }
}

/// Helper struct for delayed construction of Generics.
pub(super) struct GenericsCtor<'hir> {
    pub(super) params: SmallVec<[hir::GenericParam<'hir>; 4]>,
    where_clause: hir::WhereClause<'hir>,
    span: Span,
}

impl<'hir> GenericsCtor<'hir> {
    pub(super) fn into_generics(self, arena: &'hir Arena<'hir>) -> hir::Generics<'hir> {
        hir::Generics {
            params: arena.alloc_from_iter(self.params),
            where_clause: self.where_clause,
            span: self.span,
        }
    }
}
