use super::LoweringContext;
use super::ImplTraitContext;
use super::ImplTraitPosition;
use super::ImplTraitTypeIdVisitor;
use super::AnonymousLifetimeMode;
use super::ParamMode;

use crate::hir::{self, HirVec};
use crate::hir::ptr::P;
use crate::hir::def_id::DefId;
use crate::hir::def::{Res, DefKind};
use crate::util::nodemap::NodeMap;

use rustc_data_structures::thin_vec::ThinVec;

use std::collections::BTreeSet;
use smallvec::SmallVec;
use syntax::attr;
use syntax::ast::*;
use syntax::visit::{self, Visitor};
use syntax::expand::SpecialDerives;
use syntax::source_map::{respan, DesugaringKind, Spanned};
use syntax::symbol::{kw, sym};
use syntax_pos::Span;

pub(super) struct ItemLowerer<'tcx, 'interner> {
    pub(super) lctx: &'tcx mut LoweringContext<'interner>,
}

impl<'tcx, 'interner> ItemLowerer<'tcx, 'interner> {
    fn with_trait_impl_ref<F>(&mut self, trait_impl_ref: &Option<TraitRef>, f: F)
    where
        F: FnOnce(&mut Self),
    {
        let old = self.lctx.is_in_trait_impl;
        self.lctx.is_in_trait_impl = if let &None = trait_impl_ref {
            false
        } else {
            true
        };
        f(self);
        self.lctx.is_in_trait_impl = old;
    }
}

impl<'tcx, 'interner> Visitor<'tcx> for ItemLowerer<'tcx, 'interner> {
    fn visit_mod(&mut self, m: &'tcx Mod, _s: Span, _attrs: &[Attribute], n: NodeId) {
        let hir_id = self.lctx.lower_node_id(n);

        self.lctx.modules.insert(hir_id, hir::ModuleItems {
            items: BTreeSet::new(),
            trait_items: BTreeSet::new(),
            impl_items: BTreeSet::new(),
        });

        let old = self.lctx.current_module;
        self.lctx.current_module = hir_id;
        visit::walk_mod(self, m);
        self.lctx.current_module = old;
    }

    fn visit_item(&mut self, item: &'tcx Item) {
        let mut item_hir_id = None;
        self.lctx.with_hir_id_owner(item.id, |lctx| {
            lctx.without_in_scope_lifetime_defs(|lctx| {
                if let Some(hir_item) = lctx.lower_item(item) {
                    item_hir_id = Some(hir_item.hir_id);
                    lctx.insert_item(hir_item);
                }
            })
        });

        if let Some(hir_id) = item_hir_id {
            self.lctx.with_parent_item_lifetime_defs(hir_id, |this| {
                let this = &mut ItemLowerer { lctx: this };
                if let ItemKind::Impl(.., ref opt_trait_ref, _, _) = item.kind {
                    this.with_trait_impl_ref(opt_trait_ref, |this| {
                        visit::walk_item(this, item)
                    });
                } else {
                    visit::walk_item(this, item);
                }
            });
        }
    }

    fn visit_trait_item(&mut self, item: &'tcx TraitItem) {
        self.lctx.with_hir_id_owner(item.id, |lctx| {
            let hir_item = lctx.lower_trait_item(item);
            let id = hir::TraitItemId { hir_id: hir_item.hir_id };
            lctx.trait_items.insert(id, hir_item);
            lctx.modules.get_mut(&lctx.current_module).unwrap().trait_items.insert(id);
        });

        visit::walk_trait_item(self, item);
    }

    fn visit_impl_item(&mut self, item: &'tcx ImplItem) {
        self.lctx.with_hir_id_owner(item.id, |lctx| {
            let hir_item = lctx.lower_impl_item(item);
            let id = hir::ImplItemId { hir_id: hir_item.hir_id };
            lctx.impl_items.insert(id, hir_item);
            lctx.modules.get_mut(&lctx.current_module).unwrap().impl_items.insert(id);
        });
        visit::walk_impl_item(self, item);
    }
}

impl LoweringContext<'_> {
    // Same as the method above, but accepts `hir::GenericParam`s
    // instead of `ast::GenericParam`s.
    // This should only be used with generics that have already had their
    // in-band lifetimes added. In practice, this means that this function is
    // only used when lowering a child item of a trait or impl.
    fn with_parent_item_lifetime_defs<T>(
        &mut self,
        parent_hir_id: hir::HirId,
        f: impl FnOnce(&mut LoweringContext<'_>) -> T,
    ) -> T {
        let old_len = self.in_scope_lifetimes.len();

        let parent_generics = match self.items.get(&parent_hir_id).unwrap().kind {
            hir::ItemKind::Impl(_, _, _, ref generics, ..)
            | hir::ItemKind::Trait(_, _, ref generics, ..) => {
                &generics.params[..]
            }
            _ => &[],
        };
        let lt_def_names = parent_generics.iter().filter_map(|param| match param.kind {
            hir::GenericParamKind::Lifetime { .. } => Some(param.name.modern()),
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
    fn without_in_scope_lifetime_defs<T>(
        &mut self,
        f: impl FnOnce(&mut LoweringContext<'_>) -> T,
    ) -> T {
        let old_in_scope_lifetimes = std::mem::replace(&mut self.in_scope_lifetimes, vec![]);

        // this vector is only used when walking over impl headers,
        // input types, and the like, and should not be non-empty in
        // between items
        assert!(self.lifetimes_to_define.is_empty());

        let res = f(self);

        assert!(self.in_scope_lifetimes.is_empty());
        self.in_scope_lifetimes = old_in_scope_lifetimes;

        res
    }

    pub(super) fn lower_mod(&mut self, m: &Mod) -> hir::Mod {
        hir::Mod {
            inner: m.inner,
            item_ids: m.items.iter().flat_map(|x| self.lower_item_id(x)).collect(),
        }
    }

    pub(super) fn lower_item_id(&mut self, i: &Item) -> SmallVec<[hir::ItemId; 1]> {
        let node_ids = match i.kind {
            ItemKind::Use(ref use_tree) => {
                let mut vec = smallvec![i.id];
                self.lower_item_id_use_tree(use_tree, i.id, &mut vec);
                vec
            }
            ItemKind::MacroDef(..) => SmallVec::new(),
            ItemKind::Fn(..) |
            ItemKind::Impl(.., None, _, _) => smallvec![i.id],
            ItemKind::Static(ref ty, ..) => {
                let mut ids = smallvec![i.id];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            ItemKind::Const(ref ty, ..) => {
                let mut ids = smallvec![i.id];
                if self.sess.features_untracked().impl_trait_in_bindings {
                    let mut visitor = ImplTraitTypeIdVisitor { ids: &mut ids };
                    visitor.visit_ty(ty);
                }
                ids
            },
            _ => smallvec![i.id],
        };

        node_ids.into_iter().map(|node_id| hir::ItemId {
            id: self.allocate_hir_id_counter(node_id)
        }).collect()
    }

    fn lower_item_id_use_tree(
        &mut self,
        tree: &UseTree,
        base_id: NodeId,
        vec: &mut SmallVec<[NodeId; 1]>
    ) {
        match tree.kind {
            UseTreeKind::Nested(ref nested_vec) => for &(ref nested, id) in nested_vec {
                vec.push(id);
                self.lower_item_id_use_tree(nested, id, vec);
            },
            UseTreeKind::Glob => {}
            UseTreeKind::Simple(_, id1, id2) => {
                for (_, &id) in self.expect_full_res_from_use(base_id)
                                    .skip(1)
                                    .zip([id1, id2].iter())
                {
                    vec.push(id);
                }
            },
        }
    }

    pub fn lower_item(&mut self, i: &Item) -> Option<hir::Item> {
        let mut ident = i.ident;
        let mut vis = self.lower_visibility(&i.vis, None);
        let mut attrs = self.lower_attrs_extendable(&i.attrs);
        if self.resolver.has_derives(i.id, SpecialDerives::PARTIAL_EQ | SpecialDerives::EQ) {
            // Add `#[structural_match]` if the item derived both `PartialEq` and `Eq`.
            let ident = Ident::new(sym::structural_match, i.span);
            attrs.push(attr::mk_attr_outer(attr::mk_word_item(ident)));
        }
        let attrs = attrs.into();

        if let ItemKind::MacroDef(ref def) = i.kind {
            if !def.legacy || attr::contains_name(&i.attrs, sym::macro_export) {
                let body = self.lower_token_stream(def.stream());
                let hir_id = self.lower_node_id(i.id);
                self.exported_macros.push(hir::MacroDef {
                    name: ident.name,
                    vis,
                    attrs,
                    hir_id,
                    span: i.span,
                    body,
                    legacy: def.legacy,
                });
            } else {
                self.non_exported_macro_attrs.extend(attrs.into_iter());
            }
            return None;
        }

        let kind = self.lower_item_kind(i.id, &mut ident, &attrs, &mut vis, &i.kind);

        Some(hir::Item {
            hir_id: self.lower_node_id(i.id),
            ident,
            attrs,
            kind,
            vis,
            span: i.span,
        })
    }

    fn lower_item_kind(
        &mut self,
        id: NodeId,
        ident: &mut Ident,
        attrs: &hir::HirVec<Attribute>,
        vis: &mut hir::Visibility,
        i: &ItemKind,
    ) -> hir::ItemKind {
        match *i {
            ItemKind::ExternCrate(orig_name) => hir::ItemKind::ExternCrate(orig_name),
            ItemKind::Use(ref use_tree) => {
                // Start with an empty prefix.
                let prefix = Path {
                    segments: vec![],
                    span: use_tree.span,
                };

                self.lower_use_tree(use_tree, &prefix, id, vis, ident, attrs)
            }
            ItemKind::Static(ref t, m, ref e) => {
                hir::ItemKind::Static(
                    self.lower_ty(
                        t,
                        if self.sess.features_untracked().impl_trait_in_bindings {
                            ImplTraitContext::OpaqueTy(None)
                        } else {
                            ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                        }
                    ),
                    self.lower_mutability(m),
                    self.lower_const_body(e),
                )
            }
            ItemKind::Const(ref t, ref e) => {
                hir::ItemKind::Const(
                    self.lower_ty(
                        t,
                        if self.sess.features_untracked().impl_trait_in_bindings {
                            ImplTraitContext::OpaqueTy(None)
                        } else {
                            ImplTraitContext::Disallowed(ImplTraitPosition::Binding)
                        }
                    ),
                    self.lower_const_body(e)
                )
            }
            ItemKind::Fn(ref decl, header, ref generics, ref body) => {
                let fn_def_id = self.resolver.definitions().local_def_id(id);
                self.with_new_scopes(|this| {
                    this.current_item = Some(ident.span);

                    // Note: we don't need to change the return type from `T` to
                    // `impl Future<Output = T>` here because lower_body
                    // only cares about the input argument patterns in the function
                    // declaration (decl), not the return types.
                    let body_id = this.lower_maybe_async_body(&decl, header.asyncness.node, body);

                    let (generics, fn_decl) = this.add_in_band_defs(
                        generics,
                        fn_def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, idty| this.lower_fn_decl(
                            &decl,
                            Some((fn_def_id, idty)),
                            true,
                            header.asyncness.node.opt_return_id()
                        ),
                    );

                    hir::ItemKind::Fn(
                        fn_decl,
                        this.lower_fn_header(header),
                        generics,
                        body_id,
                    )
                })
            }
            ItemKind::Mod(ref m) => hir::ItemKind::Mod(self.lower_mod(m)),
            ItemKind::ForeignMod(ref nm) => hir::ItemKind::ForeignMod(self.lower_foreign_mod(nm)),
            ItemKind::GlobalAsm(ref ga) => hir::ItemKind::GlobalAsm(self.lower_global_asm(ga)),
            ItemKind::TyAlias(ref t, ref generics) => hir::ItemKind::TyAlias(
                self.lower_ty(t, ImplTraitContext::disallowed()),
                self.lower_generics(generics, ImplTraitContext::disallowed()),
            ),
            ItemKind::OpaqueTy(ref b, ref generics) => hir::ItemKind::OpaqueTy(
                hir::OpaqueTy {
                    generics: self.lower_generics(generics,
                        ImplTraitContext::OpaqueTy(None)),
                    bounds: self.lower_param_bounds(b,
                        ImplTraitContext::OpaqueTy(None)),
                    impl_trait_fn: None,
                    origin: hir::OpaqueTyOrigin::TypeAlias,
                },
            ),
            ItemKind::Enum(ref enum_definition, ref generics) => {
                hir::ItemKind::Enum(
                    hir::EnumDef {
                        variants: enum_definition
                            .variants
                            .iter()
                            .map(|x| self.lower_variant(x))
                            .collect(),
                    },
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            },
            ItemKind::Struct(ref struct_def, ref generics) => {
                let struct_def = self.lower_variant_data(struct_def);
                hir::ItemKind::Struct(
                    struct_def,
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            }
            ItemKind::Union(ref vdata, ref generics) => {
                let vdata = self.lower_variant_data(vdata);
                hir::ItemKind::Union(
                    vdata,
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                )
            }
            ItemKind::Impl(
                unsafety,
                polarity,
                defaultness,
                ref ast_generics,
                ref trait_ref,
                ref ty,
                ref impl_items,
            ) => {
                let def_id = self.resolver.definitions().local_def_id(id);

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
                let lowered_trait_impl_id = self.lower_node_id(id);
                let (generics, (trait_ref, lowered_ty)) = self.add_in_band_defs(
                    ast_generics,
                    def_id,
                    AnonymousLifetimeMode::CreateParameter,
                    |this, _| {
                        let trait_ref = trait_ref.as_ref().map(|trait_ref| {
                            this.lower_trait_ref(trait_ref, ImplTraitContext::disallowed())
                        });

                        if let Some(ref trait_ref) = trait_ref {
                            if let Res::Def(DefKind::Trait, def_id) = trait_ref.path.res {
                                this.trait_impls.entry(def_id).or_default().push(
                                    lowered_trait_impl_id);
                            }
                        }

                        let lowered_ty = this.lower_ty(ty, ImplTraitContext::disallowed());

                        (trait_ref, lowered_ty)
                    },
                );

                let new_impl_items = self.with_in_scope_lifetime_defs(
                    &ast_generics.params,
                    |this| {
                        impl_items
                            .iter()
                            .map(|item| this.lower_impl_item_ref(item))
                            .collect()
                    },
                );

                hir::ItemKind::Impl(
                    self.lower_unsafety(unsafety),
                    self.lower_impl_polarity(polarity),
                    self.lower_defaultness(defaultness, true /* [1] */),
                    generics,
                    trait_ref,
                    lowered_ty,
                    new_impl_items,
                )
            }
            ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref items) => {
                let bounds = self.lower_param_bounds(bounds, ImplTraitContext::disallowed());
                let items = items
                    .iter()
                    .map(|item| self.lower_trait_item_ref(item))
                    .collect();
                hir::ItemKind::Trait(
                    self.lower_is_auto(is_auto),
                    self.lower_unsafety(unsafety),
                    self.lower_generics(generics, ImplTraitContext::disallowed()),
                    bounds,
                    items,
                )
            }
            ItemKind::TraitAlias(ref generics, ref bounds) => hir::ItemKind::TraitAlias(
                self.lower_generics(generics, ImplTraitContext::disallowed()),
                self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
            ),
            ItemKind::MacroDef(..)
            | ItemKind::Mac(..) => bug!("`TyMac` should have been expanded by now"),
        }

        // [1] `defaultness.has_value()` is never called for an `impl`, always `true` in order to
        //     not cause an assertion failure inside the `lower_defaultness` function.
    }

    fn lower_use_tree(
        &mut self,
        tree: &UseTree,
        prefix: &Path,
        id: NodeId,
        vis: &mut hir::Visibility,
        ident: &mut Ident,
        attrs: &hir::HirVec<Attribute>,
    ) -> hir::ItemKind {
        debug!("lower_use_tree(tree={:?})", tree);
        debug!("lower_use_tree: vis = {:?}", vis);

        let path = &tree.prefix;
        let segments = prefix
            .segments
            .iter()
            .chain(path.segments.iter())
            .cloned()
            .collect();

        match tree.kind {
            UseTreeKind::Simple(rename, id1, id2) => {
                *ident = tree.ident();

                // First, apply the prefix to the path.
                let mut path = Path {
                    segments,
                    span: path.span,
                };

                // Correctly resolve `self` imports.
                if path.segments.len() > 1
                    && path.segments.last().unwrap().ident.name == kw::SelfLower
                {
                    let _ = path.segments.pop();
                    if rename.is_none() {
                        *ident = path.segments.last().unwrap().ident;
                    }
                }

                let mut resolutions = self.expect_full_res_from_use(id);
                // We want to return *something* from this function, so hold onto the first item
                // for later.
                let ret_res = self.lower_res(resolutions.next().unwrap_or(Res::Err));

                // Here, we are looping over namespaces, if they exist for the definition
                // being imported. We only handle type and value namespaces because we
                // won't be dealing with macros in the rest of the compiler.
                // Essentially a single `use` which imports two names is desugared into
                // two imports.
                for (res, &new_node_id) in resolutions.zip([id1, id2].iter()) {
                    let ident = *ident;
                    let mut path = path.clone();
                    for seg in &mut path.segments {
                        seg.id = self.sess.next_node_id();
                    }
                    let span = path.span;

                    self.with_hir_id_owner(new_node_id, |this| {
                        let new_id = this.lower_node_id(new_node_id);
                        let res = this.lower_res(res);
                        let path =
                            this.lower_path_extra(res, &path, ParamMode::Explicit, None);
                        let kind = hir::ItemKind::Use(P(path), hir::UseKind::Single);
                        let vis = this.rebuild_vis(&vis);

                        this.insert_item(
                            hir::Item {
                                hir_id: new_id,
                                ident,
                                attrs: attrs.into_iter().cloned().collect(),
                                kind,
                                vis,
                                span,
                            },
                        );
                    });
                }

                let path = P(self.lower_path_extra(ret_res, &path, ParamMode::Explicit, None));
                hir::ItemKind::Use(path, hir::UseKind::Single)
            }
            UseTreeKind::Glob => {
                let path = P(self.lower_path(
                    id,
                    &Path {
                        segments,
                        span: path.span,
                    },
                    ParamMode::Explicit,
                ));
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

                let prefix = Path {
                    segments,
                    span: prefix.span.to(path.span),
                };

                // Add all the nested `PathListItem`s to the HIR.
                for &(ref use_tree, id) in trees {
                    let new_hir_id = self.lower_node_id(id);

                    let mut prefix = prefix.clone();

                    // Give the segments new node-ids since they are being cloned.
                    for seg in &mut prefix.segments {
                        seg.id = self.sess.next_node_id();
                    }

                    // Each `use` import is an item and thus are owners of the
                    // names in the path. Up to this point the nested import is
                    // the current owner, since we want each desugared import to
                    // own its own names, we have to adjust the owner before
                    // lowering the rest of the import.
                    self.with_hir_id_owner(id, |this| {
                        let mut vis = this.rebuild_vis(&vis);
                        let mut ident = *ident;

                        let kind = this.lower_use_tree(use_tree,
                                                       &prefix,
                                                       id,
                                                       &mut vis,
                                                       &mut ident,
                                                       attrs);

                        this.insert_item(
                            hir::Item {
                                hir_id: new_hir_id,
                                ident,
                                attrs: attrs.into_iter().cloned().collect(),
                                kind,
                                vis,
                                span: use_tree.span,
                            },
                        );
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
                    hir::VisibilityKind::Public |
                    hir::VisibilityKind::Crate(_) |
                    hir::VisibilityKind::Inherited => {
                        *vis = respan(prefix.span.shrink_to_lo(), hir::VisibilityKind::Inherited);
                    }
                    hir::VisibilityKind::Restricted { .. } => {
                        // Do nothing here, as described in the comment on the match.
                    }
                }

                let res = self.expect_full_res_from_use(id).next().unwrap_or(Res::Err);
                let res = self.lower_res(res);
                let path = P(self.lower_path_extra(res, &prefix, ParamMode::Explicit, None));
                hir::ItemKind::Use(path, hir::UseKind::ListStem)
            }
        }
    }

    /// Paths like the visibility path in `pub(super) use foo::{bar, baz}` are repeated
    /// many times in the HIR tree; for each occurrence, we need to assign distinct
    /// `NodeId`s. (See, e.g., #56128.)
    fn rebuild_use_path(&mut self, path: &hir::Path) -> hir::Path {
        debug!("rebuild_use_path(path = {:?})", path);
        let segments = path.segments.iter().map(|seg| hir::PathSegment {
            ident: seg.ident,
            hir_id: seg.hir_id.map(|_| self.next_id()),
            res: seg.res,
            args: None,
            infer_args: seg.infer_args,
        }).collect();
        hir::Path {
            span: path.span,
            res: path.res,
            segments,
        }
    }

    fn rebuild_vis(&mut self, vis: &hir::Visibility) -> hir::Visibility {
        let vis_kind = match vis.node {
            hir::VisibilityKind::Public => hir::VisibilityKind::Public,
            hir::VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            hir::VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
            hir::VisibilityKind::Restricted { ref path, hir_id: _ } => {
                hir::VisibilityKind::Restricted {
                    path: P(self.rebuild_use_path(path)),
                    hir_id: self.next_id(),
                }
            }
        };
        respan(vis.span, vis_kind)
    }

    fn lower_foreign_item(&mut self, i: &ForeignItem) -> hir::ForeignItem {
        let def_id = self.resolver.definitions().local_def_id(i.id);
        hir::ForeignItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            kind: match i.kind {
                ForeignItemKind::Fn(ref fdec, ref generics) => {
                    let (generics, (fn_dec, fn_args)) = self.add_in_band_defs(
                        generics,
                        def_id,
                        AnonymousLifetimeMode::PassThrough,
                        |this, _| {
                            (
                                // Disallow `impl Trait` in foreign items.
                                this.lower_fn_decl(fdec, None, false, None),
                                this.lower_fn_params_to_names(fdec),
                            )
                        },
                    );

                    hir::ForeignItemKind::Fn(fn_dec, fn_args, generics)
                }
                ForeignItemKind::Static(ref t, m) => {
                    hir::ForeignItemKind::Static(
                        self.lower_ty(t, ImplTraitContext::disallowed()), self.lower_mutability(m))
                }
                ForeignItemKind::Ty => hir::ForeignItemKind::Type,
                ForeignItemKind::Macro(_) => panic!("macro shouldn't exist here"),
            },
            vis: self.lower_visibility(&i.vis, None),
            span: i.span,
        }
    }

    fn lower_foreign_mod(&mut self, fm: &ForeignMod) -> hir::ForeignMod {
        hir::ForeignMod {
            abi: fm.abi,
            items: fm.items
                .iter()
                .map(|x| self.lower_foreign_item(x))
                .collect(),
        }
    }

    fn lower_global_asm(&mut self, ga: &GlobalAsm) -> P<hir::GlobalAsm> {
        P(hir::GlobalAsm { asm: ga.asm })
    }

    fn lower_variant(&mut self, v: &Variant) -> hir::Variant {
        hir::Variant {
            attrs: self.lower_attrs(&v.attrs),
            data: self.lower_variant_data(&v.data),
            disr_expr: v.disr_expr.as_ref().map(|e| self.lower_anon_const(e)),
            id: self.lower_node_id(v.id),
            ident: v.ident,
            span: v.span,
        }
    }

    fn lower_variant_data(&mut self, vdata: &VariantData) -> hir::VariantData {
        match *vdata {
            VariantData::Struct(ref fields, recovered) => hir::VariantData::Struct(
                fields.iter().enumerate().map(|f| self.lower_struct_field(f)).collect(),
                recovered,
            ),
            VariantData::Tuple(ref fields, id) => {
                hir::VariantData::Tuple(
                    fields
                        .iter()
                        .enumerate()
                        .map(|f| self.lower_struct_field(f))
                        .collect(),
                    self.lower_node_id(id),
                )
            },
            VariantData::Unit(id) => {
                hir::VariantData::Unit(self.lower_node_id(id))
            },
        }
    }

    fn lower_struct_field(&mut self, (index, f): (usize, &StructField)) -> hir::StructField {
        let ty = if let TyKind::Path(ref qself, ref path) = f.ty.kind {
            let t = self.lower_path_ty(
                &f.ty,
                qself,
                path,
                ParamMode::ExplicitNamed, // no `'_` in declarations (Issue #61124)
                ImplTraitContext::disallowed()
            );
            P(t)
        } else {
            self.lower_ty(&f.ty, ImplTraitContext::disallowed())
        };
        hir::StructField {
            span: f.span,
            hir_id: self.lower_node_id(f.id),
            ident: match f.ident {
                Some(ident) => ident,
                // FIXME(jseyfried): positional field hygiene.
                None => Ident::new(sym::integer(index), f.span),
            },
            vis: self.lower_visibility(&f.vis, None),
            ty,
            attrs: self.lower_attrs(&f.attrs),
        }
    }

    fn lower_trait_item(&mut self, i: &TraitItem) -> hir::TraitItem {
        let trait_item_def_id = self.resolver.definitions().local_def_id(i.id);

        let (generics, kind) = match i.kind {
            TraitItemKind::Const(ref ty, ref default) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::TraitItemKind::Const(
                    self.lower_ty(ty, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_const_body(x)),
                ),
            ),
            TraitItemKind::Method(ref sig, None) => {
                let names = self.lower_fn_params_to_names(&sig.decl);
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    trait_item_def_id,
                    false,
                    None,
                );
                (generics, hir::TraitItemKind::Method(sig, hir::TraitMethod::Required(names)))
            }
            TraitItemKind::Method(ref sig, Some(ref body)) => {
                let body_id = self.lower_fn_body_block(&sig.decl, body);
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    trait_item_def_id,
                    false,
                    None,
                );
                (generics, hir::TraitItemKind::Method(sig, hir::TraitMethod::Provided(body_id)))
            }
            TraitItemKind::Type(ref bounds, ref default) => {
                let generics = self.lower_generics(&i.generics, ImplTraitContext::disallowed());
                let kind = hir::TraitItemKind::Type(
                    self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
                    default
                        .as_ref()
                        .map(|x| self.lower_ty(x, ImplTraitContext::disallowed())),
                );

                (generics, kind)
            },
            TraitItemKind::Macro(..) => bug!("macro item shouldn't exist at this point"),
        };

        hir::TraitItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            generics,
            kind,
            span: i.span,
        }
    }

    fn lower_trait_item_ref(&mut self, i: &TraitItem) -> hir::TraitItemRef {
        let (kind, has_default) = match i.kind {
            TraitItemKind::Const(_, ref default) => {
                (hir::AssocItemKind::Const, default.is_some())
            }
            TraitItemKind::Type(_, ref default) => {
                (hir::AssocItemKind::Type, default.is_some())
            }
            TraitItemKind::Method(ref sig, ref default) => (
                hir::AssocItemKind::Method {
                    has_self: sig.decl.has_self(),
                },
                default.is_some(),
            ),
            TraitItemKind::Macro(..) => unimplemented!(),
        };
        hir::TraitItemRef {
            id: hir::TraitItemId { hir_id: self.lower_node_id(i.id) },
            ident: i.ident,
            span: i.span,
            defaultness: self.lower_defaultness(Defaultness::Default, has_default),
            kind,
        }
    }

    fn lower_impl_item(&mut self, i: &ImplItem) -> hir::ImplItem {
        let impl_item_def_id = self.resolver.definitions().local_def_id(i.id);

        let (generics, kind) = match i.kind {
            ImplItemKind::Const(ref ty, ref expr) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::Const(
                    self.lower_ty(ty, ImplTraitContext::disallowed()),
                    self.lower_const_body(expr),
                ),
            ),
            ImplItemKind::Method(ref sig, ref body) => {
                self.current_item = Some(i.span);
                let body_id = self.lower_maybe_async_body(
                    &sig.decl, sig.header.asyncness.node, body
                );
                let impl_trait_return_allow = !self.is_in_trait_impl;
                let (generics, sig) = self.lower_method_sig(
                    &i.generics,
                    sig,
                    impl_item_def_id,
                    impl_trait_return_allow,
                    sig.header.asyncness.node.opt_return_id(),
                );

                (generics, hir::ImplItemKind::Method(sig, body_id))
            }
            ImplItemKind::TyAlias(ref ty) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::TyAlias(self.lower_ty(ty, ImplTraitContext::disallowed())),
            ),
            ImplItemKind::OpaqueTy(ref bounds) => (
                self.lower_generics(&i.generics, ImplTraitContext::disallowed()),
                hir::ImplItemKind::OpaqueTy(
                    self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
                ),
            ),
            ImplItemKind::Macro(..) => bug!("`TyMac` should have been expanded by now"),
        };

        hir::ImplItem {
            hir_id: self.lower_node_id(i.id),
            ident: i.ident,
            attrs: self.lower_attrs(&i.attrs),
            generics,
            vis: self.lower_visibility(&i.vis, None),
            defaultness: self.lower_defaultness(i.defaultness, true /* [1] */),
            kind,
            span: i.span,
        }

        // [1] since `default impl` is not yet implemented, this is always true in impls
    }

    fn lower_impl_item_ref(&mut self, i: &ImplItem) -> hir::ImplItemRef {
        hir::ImplItemRef {
            id: hir::ImplItemId { hir_id: self.lower_node_id(i.id) },
            ident: i.ident,
            span: i.span,
            vis: self.lower_visibility(&i.vis, Some(i.id)),
            defaultness: self.lower_defaultness(i.defaultness, true /* [1] */),
            kind: match i.kind {
                ImplItemKind::Const(..) => hir::AssocItemKind::Const,
                ImplItemKind::TyAlias(..) => hir::AssocItemKind::Type,
                ImplItemKind::OpaqueTy(..) => hir::AssocItemKind::OpaqueTy,
                ImplItemKind::Method(ref sig, _) => hir::AssocItemKind::Method {
                    has_self: sig.decl.has_self(),
                },
                ImplItemKind::Macro(..) => unimplemented!(),
            },
        }

        // [1] since `default impl` is not yet implemented, this is always true in impls
    }

    /// If an `explicit_owner` is given, this method allocates the `HirId` in
    /// the address space of that item instead of the item currently being
    /// lowered. This can happen during `lower_impl_item_ref()` where we need to
    /// lower a `Visibility` value although we haven't lowered the owning
    /// `ImplItem` in question yet.
    fn lower_visibility(
        &mut self,
        v: &Visibility,
        explicit_owner: Option<NodeId>,
    ) -> hir::Visibility {
        let node = match v.node {
            VisibilityKind::Public => hir::VisibilityKind::Public,
            VisibilityKind::Crate(sugar) => hir::VisibilityKind::Crate(sugar),
            VisibilityKind::Restricted { ref path, id } => {
                debug!("lower_visibility: restricted path id = {:?}", id);
                let lowered_id = if let Some(owner) = explicit_owner {
                    self.lower_node_id_with_owner(id, owner)
                } else {
                    self.lower_node_id(id)
                };
                let res = self.expect_full_res(id);
                let res = self.lower_res(res);
                hir::VisibilityKind::Restricted {
                    path: P(self.lower_path_extra(
                        res,
                        path,
                        ParamMode::Explicit,
                        explicit_owner,
                    )),
                    hir_id: lowered_id,
                }
            },
            VisibilityKind::Inherited => hir::VisibilityKind::Inherited,
        };
        respan(v.span, node)
    }

    fn lower_defaultness(&self, d: Defaultness, has_value: bool) -> hir::Defaultness {
        match d {
            Defaultness::Default => hir::Defaultness::Default {
                has_value: has_value,
            },
            Defaultness::Final => {
                assert!(has_value);
                hir::Defaultness::Final
            }
        }
    }

    fn lower_impl_polarity(&mut self, i: ImplPolarity) -> hir::ImplPolarity {
        match i {
            ImplPolarity::Positive => hir::ImplPolarity::Positive,
            ImplPolarity::Negative => hir::ImplPolarity::Negative,
        }
    }

    fn record_body(&mut self, params: HirVec<hir::Param>, value: hir::Expr) -> hir::BodyId {
        let body = hir::Body {
            generator_kind: self.generator_kind,
            params,
            value,
        };
        let id = body.id();
        self.bodies.insert(id, body);
        id
    }

    fn lower_body(
        &mut self,
        f: impl FnOnce(&mut LoweringContext<'_>) -> (HirVec<hir::Param>, hir::Expr),
    ) -> hir::BodyId {
        let prev_gen_kind = self.generator_kind.take();
        let (parameters, result) = f(self);
        let body_id = self.record_body(parameters, result);
        self.generator_kind = prev_gen_kind;
        body_id
    }

    fn lower_param(&mut self, param: &Param) -> hir::Param {
        hir::Param {
            attrs: self.lower_attrs(&param.attrs),
            hir_id: self.lower_node_id(param.id),
            pat: self.lower_pat(&param.pat),
            span: param.span,
        }
    }

    pub(super) fn lower_fn_body(
        &mut self,
        decl: &FnDecl,
        body: impl FnOnce(&mut LoweringContext<'_>) -> hir::Expr,
    ) -> hir::BodyId {
        self.lower_body(|this| (
            decl.inputs.iter().map(|x| this.lower_param(x)).collect(),
            body(this),
        ))
    }

    fn lower_fn_body_block(&mut self, decl: &FnDecl, body: &Block) -> hir::BodyId {
        self.lower_fn_body(decl, |this| this.lower_block_expr(body))
    }

    pub(super) fn lower_const_body(&mut self, expr: &Expr) -> hir::BodyId {
        self.lower_body(|this| (hir_vec![], this.lower_expr(expr)))
    }

    fn lower_maybe_async_body(
        &mut self,
        decl: &FnDecl,
        asyncness: IsAsync,
        body: &Block,
    ) -> hir::BodyId {
        let closure_id = match asyncness {
            IsAsync::Async { closure_id, .. } => closure_id,
            IsAsync::NotAsync => return self.lower_fn_body_block(decl, body),
        };

        self.lower_body(|this| {
            let mut parameters: Vec<hir::Param> = Vec::new();
            let mut statements: Vec<hir::Stmt> = Vec::new();

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
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, ident, _) =>
                        (ident, true),
                    _ => {
                        // Replace the ident for bindings that aren't simple.
                        let name = format!("__arg{}", index);
                        let ident = Ident::from_str(&name);

                        (ident, false)
                    },
                };

                let desugared_span =
                    this.mark_span_with_reason(DesugaringKind::Async, span, None);

                // Construct a parameter representing `__argN: <ty>` to replace the parameter of the
                // async function.
                //
                // If this is the simple case, this parameter will end up being the same as the
                // original parameter, but with a different pattern id.
                let mut stmt_attrs = ThinVec::new();
                stmt_attrs.extend(parameter.attrs.iter().cloned());
                let (new_parameter_pat, new_parameter_id) = this.pat_ident(desugared_span, ident);
                let new_parameter = hir::Param {
                    attrs: parameter.attrs,
                    hir_id: parameter.hir_id,
                    pat: new_parameter_pat,
                    span: parameter.span,
                };


                if is_simple_parameter {
                    // If this is the simple case, then we only insert one statement that is
                    // `let <pat> = <pat>;`. We re-use the original argument's pattern so that
                    // `HirId`s are densely assigned.
                    let expr = this.expr_ident(desugared_span, ident, new_parameter_id);
                    let stmt = this.stmt_let_pat(
                        stmt_attrs,
                        desugared_span,
                        Some(P(expr)),
                        parameter.pat,
                        hir::LocalSource::AsyncFn
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
                        desugared_span, ident, hir::BindingAnnotation::Mutable);
                    let move_expr = this.expr_ident(desugared_span, ident, new_parameter_id);
                    let move_stmt = this.stmt_let_pat(
                        ThinVec::new(),
                        desugared_span,
                        Some(P(move_expr)),
                        move_pat,
                        hir::LocalSource::AsyncFn
                    );

                    // Construct the `let <pat> = __argN;` statement. We re-use the original
                    // parameter's pattern so that `HirId`s are densely assigned.
                    let pattern_expr = this.expr_ident(desugared_span, ident, move_id);
                    let pattern_stmt = this.stmt_let_pat(
                        stmt_attrs,
                        desugared_span,
                        Some(P(pattern_expr)),
                        parameter.pat,
                        hir::LocalSource::AsyncFn
                    );

                    statements.push(move_stmt);
                    statements.push(pattern_stmt);
                };

                parameters.push(new_parameter);
            }

            let async_expr = this.make_async_expr(
                CaptureBy::Value,
                closure_id,
                None,
                body.span,
                hir::AsyncGeneratorKind::Fn,
                |this| {
                    // Create a block from the user's function body:
                    let user_body = this.lower_block_expr(body);

                    // Transform into `drop-temps { <user-body> }`, an expression:
                    let desugared_span = this.mark_span_with_reason(
                        DesugaringKind::Async,
                        user_body.span,
                        None,
                    );
                    let user_body = this.expr_drop_temps(
                        desugared_span,
                        P(user_body),
                        ThinVec::new(),
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
                        statements.into(),
                        Some(P(user_body)),
                    );
                    this.expr_block(P(body), ThinVec::new())
                });
            (HirVec::from(parameters), this.expr(body.span, async_expr, ThinVec::new()))
        })
    }

    fn lower_method_sig(
        &mut self,
        generics: &Generics,
        sig: &MethodSig,
        fn_def_id: DefId,
        impl_trait_return_allow: bool,
        is_async: Option<NodeId>,
    ) -> (hir::Generics, hir::MethodSig) {
        let header = self.lower_fn_header(sig.header);
        let (generics, decl) = self.add_in_band_defs(
            generics,
            fn_def_id,
            AnonymousLifetimeMode::PassThrough,
            |this, idty| this.lower_fn_decl(
                &sig.decl,
                Some((fn_def_id, idty)),
                impl_trait_return_allow,
                is_async,
            ),
        );
        (generics, hir::MethodSig { header, decl })
    }

    fn lower_is_auto(&mut self, a: IsAuto) -> hir::IsAuto {
        match a {
            IsAuto::Yes => hir::IsAuto::Yes,
            IsAuto::No => hir::IsAuto::No,
        }
    }

    fn lower_fn_header(&mut self, h: FnHeader) -> hir::FnHeader {
        hir::FnHeader {
            unsafety: self.lower_unsafety(h.unsafety),
            asyncness: self.lower_asyncness(h.asyncness.node),
            constness: self.lower_constness(h.constness),
            abi: h.abi,
        }
    }

    pub(super) fn lower_unsafety(&mut self, u: Unsafety) -> hir::Unsafety {
        match u {
            Unsafety::Unsafe => hir::Unsafety::Unsafe,
            Unsafety::Normal => hir::Unsafety::Normal,
        }
    }

    fn lower_constness(&mut self, c: Spanned<Constness>) -> hir::Constness {
        match c.node {
            Constness::Const => hir::Constness::Const,
            Constness::NotConst => hir::Constness::NotConst,
        }
    }

    fn lower_asyncness(&mut self, a: IsAsync) -> hir::IsAsync {
        match a {
            IsAsync::Async { .. } => hir::IsAsync::Async,
            IsAsync::NotAsync => hir::IsAsync::NotAsync,
        }
    }

    pub(super) fn lower_generics(
        &mut self,
        generics: &Generics,
        itctx: ImplTraitContext<'_>)
        -> hir::Generics
    {
        // Collect `?Trait` bounds in where clause and move them to parameter definitions.
        // FIXME: this could probably be done with less rightward drift. It also looks like two
        // control paths where `report_error` is called are the only paths that advance to after the
        // match statement, so the error reporting could probably just be moved there.
        let mut add_bounds: NodeMap<Vec<_>> = Default::default();
        for pred in &generics.where_clause.predicates {
            if let WherePredicate::BoundPredicate(ref bound_pred) = *pred {
                'next_bound: for bound in &bound_pred.bounds {
                    if let GenericBound::Trait(_, TraitBoundModifier::Maybe) = *bound {
                        let report_error = |this: &mut Self| {
                            this.diagnostic().span_err(
                                bound_pred.bounded_ty.span,
                                "`?Trait` bounds are only permitted at the \
                                 point where a type parameter is declared",
                            );
                        };
                        // Check if the where clause type is a plain type parameter.
                        match bound_pred.bounded_ty.kind {
                            TyKind::Path(None, ref path)
                                if path.segments.len() == 1
                                    && bound_pred.bound_generic_params.is_empty() =>
                            {
                                if let Some(Res::Def(DefKind::TyParam, def_id)) = self.resolver
                                    .get_partial_res(bound_pred.bounded_ty.id)
                                    .map(|d| d.base_res())
                                {
                                    if let Some(node_id) =
                                        self.resolver.definitions().as_local_node_id(def_id)
                                    {
                                        for param in &generics.params {
                                            match param.kind {
                                                GenericParamKind::Type { .. } => {
                                                    if node_id == param.id {
                                                        add_bounds.entry(param.id)
                                                            .or_default()
                                                            .push(bound.clone());
                                                        continue 'next_bound;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                report_error(self)
                            }
                            _ => report_error(self),
                        }
                    }
                }
            }
        }

        hir::Generics {
            params: self.lower_generic_params(&generics.params, &add_bounds, itctx),
            where_clause: self.lower_where_clause(&generics.where_clause),
            span: generics.span,
        }
    }

    fn lower_where_clause(&mut self, wc: &WhereClause) -> hir::WhereClause {
        self.with_anonymous_lifetime_mode(
            AnonymousLifetimeMode::ReportError,
            |this| {
                hir::WhereClause {
                    predicates: wc.predicates
                        .iter()
                        .map(|predicate| this.lower_where_predicate(predicate))
                        .collect(),
                    span: wc.span,
                }
            },
        )
    }

    fn lower_where_predicate(&mut self, pred: &WherePredicate) -> hir::WherePredicate {
        match *pred {
            WherePredicate::BoundPredicate(WhereBoundPredicate {
                ref bound_generic_params,
                ref bounded_ty,
                ref bounds,
                span,
            }) => {
                self.with_in_scope_lifetime_defs(
                    &bound_generic_params,
                    |this| {
                        hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                            bound_generic_params: this.lower_generic_params(
                                bound_generic_params,
                                &NodeMap::default(),
                                ImplTraitContext::disallowed(),
                            ),
                            bounded_ty: this.lower_ty(bounded_ty, ImplTraitContext::disallowed()),
                            bounds: bounds
                                .iter()
                                .filter_map(|bound| match *bound {
                                    // Ignore `?Trait` bounds.
                                    // They were copied into type parameters already.
                                    GenericBound::Trait(_, TraitBoundModifier::Maybe) => None,
                                    _ => Some(this.lower_param_bound(
                                        bound,
                                        ImplTraitContext::disallowed(),
                                    )),
                                })
                                .collect(),
                            span,
                        })
                    },
                )
            }
            WherePredicate::RegionPredicate(WhereRegionPredicate {
                ref lifetime,
                ref bounds,
                span,
            }) => hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                span,
                lifetime: self.lower_lifetime(lifetime),
                bounds: self.lower_param_bounds(bounds, ImplTraitContext::disallowed()),
            }),
            WherePredicate::EqPredicate(WhereEqPredicate {
                id,
                ref lhs_ty,
                ref rhs_ty,
                span,
            }) => {
                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    hir_id: self.lower_node_id(id),
                    lhs_ty: self.lower_ty(lhs_ty, ImplTraitContext::disallowed()),
                    rhs_ty: self.lower_ty(rhs_ty, ImplTraitContext::disallowed()),
                    span,
                })
            },
        }
    }
}
