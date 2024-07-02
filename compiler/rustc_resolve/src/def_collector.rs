use crate::{ImplTraitContext, Resolver};
use rustc_ast::ptr::P;
use rustc_ast::*;
use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_expand::expand::AstFragment;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use smallvec::{smallvec, SmallVec};
use tracing::debug;

pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_, '_>,
    fragment: &mut AstFragment,
    expansion: LocalExpnId,
) {
    let (parent_def, impl_trait_context) = resolver.invocation_parents[&expansion];
    fragment.mut_visit_with(&mut DefCollector {
        resolver,
        parent_def,
        expansion,
        impl_trait_context,
    });
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a, 'b, 'tcx> {
    resolver: &'a mut Resolver<'b, 'tcx>,
    parent_def: LocalDefId,
    impl_trait_context: ImplTraitContext,
    expansion: LocalExpnId,
}

impl<'a, 'b, 'tcx> DefCollector<'a, 'b, 'tcx> {
    fn create_def(
        &mut self,
        node_id: NodeId,
        name: Symbol,
        def_kind: DefKind,
        span: Span,
    ) -> LocalDefId {
        let parent_def = self.parent_def;
        debug!(
            "create_def(node_id={:?}, def_kind={:?}, parent_def={:?})",
            node_id, def_kind, parent_def
        );
        self.resolver
            .create_def(
                parent_def,
                node_id,
                name,
                def_kind,
                self.expansion.to_expn_id(),
                span.with_parent(None),
            )
            .def_id()
    }

    fn with_parent<R, F: FnOnce(&mut Self) -> R>(&mut self, parent_def: LocalDefId, f: F) -> R {
        let orig_parent_def = std::mem::replace(&mut self.parent_def, parent_def);
        let res = f(self);
        self.parent_def = orig_parent_def;
        res
    }

    fn with_impl_trait<R, F: FnOnce(&mut Self) -> R>(
        &mut self,
        impl_trait_context: ImplTraitContext,
        f: F,
    ) -> R {
        let orig_itc = std::mem::replace(&mut self.impl_trait_context, impl_trait_context);
        let res = f(self);
        self.impl_trait_context = orig_itc;
        res
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn collect_field(
        &mut self,
        mut field: FieldDef,
        index: Option<usize>,
    ) -> SmallVec<[FieldDef; 1]> {
        let index = |this: &Self| {
            index.unwrap_or_else(|| {
                let node_id = NodeId::placeholder_from_expn_id(this.expansion);
                this.resolver.placeholder_field_indices[&node_id]
            })
        };

        if field.is_placeholder {
            let old_index = self.resolver.placeholder_field_indices.insert(field.id, index(self));
            assert!(old_index.is_none(), "placeholder field index is reset for a node ID");
            self.visit_macro_invoc(field.id);
            return smallvec![field];
        } else {
            let name = field.ident.map_or_else(|| sym::integer(index(self)), |ident| ident.name);
            let def = self.create_def(field.id, name, DefKind::Field, field.span);
            self.with_parent(def, |this| {
                this.visit_anon_adt(&mut field.ty);
                mut_visit::noop_flat_map_field_def(field, this)
            })
        }
    }

    fn visit_anon_adt(&mut self, ty: &mut P<Ty>) {
        let def_kind = match &ty.kind {
            TyKind::AnonStruct(..) => DefKind::Struct,
            TyKind::AnonUnion(..) => DefKind::Union,
            _ => return,
        };
        match &ty.kind {
            TyKind::AnonStruct(node_id, _) | TyKind::AnonUnion(node_id, _) => {
                let def_id = self.create_def(*node_id, kw::Empty, def_kind, ty.span);
                self.with_parent(def_id, |this| mut_visit::noop_visit_ty(ty, this));
            }
            _ => {}
        }
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_macro_invoc(&mut self, id: NodeId) {
        let id = id.placeholder_to_expn_id();
        let old_parent =
            self.resolver.invocation_parents.insert(id, (self.parent_def, self.impl_trait_context));
        assert!(old_parent.is_none(), "parent `LocalDefId` is reset for an invocation");
    }
}

impl<'a, 'b, 'tcx> mut_visit::MutVisitor for DefCollector<'a, 'b, 'tcx> {
    fn visit_span(&mut self, span: &mut Span) {
        *span = span.with_parent(Some(self.parent_def));
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn flat_map_item(&mut self, mut i: P<Item>) -> SmallVec<[P<Item>; 1]> {
        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into, the better
        let mut opt_macro_data = None;
        let def_kind = match &i.kind {
            ItemKind::Impl(i) => DefKind::Impl { of_trait: i.of_trait.is_some() },
            ItemKind::ForeignMod(..) => DefKind::ForeignMod,
            ItemKind::Mod(..) => DefKind::Mod,
            ItemKind::Trait(..) => DefKind::Trait,
            ItemKind::TraitAlias(..) => DefKind::TraitAlias,
            ItemKind::Enum(..) => DefKind::Enum,
            ItemKind::Struct(..) => DefKind::Struct,
            ItemKind::Union(..) => DefKind::Union,
            ItemKind::ExternCrate(..) => DefKind::ExternCrate,
            ItemKind::TyAlias(..) => DefKind::TyAlias,
            ItemKind::Static(s) => DefKind::Static {
                safety: hir::Safety::Safe,
                mutability: s.mutability,
                nested: false,
            },
            ItemKind::Const(..) => DefKind::Const,
            ItemKind::Fn(..) | ItemKind::Delegation(..) => DefKind::Fn,
            ItemKind::MacroDef(..) => {
                let macro_data = self.resolver.compile_macro(&i, self.resolver.tcx.sess.edition());
                let macro_kind = macro_data.ext.macro_kind();
                opt_macro_data = Some(macro_data);
                DefKind::Macro(macro_kind)
            }
            ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
            ItemKind::Use(..) => DefKind::Use,
            ItemKind::MacCall(..) | ItemKind::DelegationMac(..) => {
                self.visit_macro_invoc(i.id);
                return smallvec![i];
            }
        };
        let def_id = self.create_def(i.id, i.ident.name, def_kind, i.span);

        if let Some(macro_data) = opt_macro_data {
            self.resolver.macro_map.insert(def_id.to_def_id(), macro_data);
        }

        self.with_parent(def_id, |this| {
            this.with_impl_trait(ImplTraitContext::Existential, |this| {
                let item = &mut *i;
                match &mut item.kind {
                    ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                        // If this is a unit or tuple-like struct, register the constructor.
                        if let Some((ctor_kind, ctor_node_id)) = CtorKind::from_ast(struct_def) {
                            this.create_def(
                                ctor_node_id,
                                kw::Empty,
                                DefKind::Ctor(CtorOf::Struct, ctor_kind),
                                item.span,
                            );
                        }
                    }
                    ItemKind::Fn(box Fn { defaultness, generics, sig, body })
                        if let Some(coroutine_kind) = sig.header.coroutine_kind =>
                    {
                        // For async functions, we need to create their inner defs inside of a
                        // closure to match their desugared representation. Besides that,
                        // we must mirror everything that `noop_flat_map_item` below does.
                        mut_visit::visit_attrs(&mut item.attrs, this);
                        this.visit_vis(&mut item.vis);
                        this.visit_ident(&mut item.ident);
                        this.visit_span(&mut item.span);
                        mut_visit::visit_defaultness(defaultness, this);
                        this.visit_generics(generics);
                        mut_visit::visit_fn_sig(sig, this);
                        // If this async fn has no body (i.e. it's an async fn signature in a trait)
                        // then the closure_def will never be used, and we should avoid generating a
                        // def-id for it.
                        if let Some(body) = body {
                            let closure_def = this.create_def(
                                coroutine_kind.closure_id(),
                                kw::Empty,
                                DefKind::Closure,
                                item.span,
                            );
                            this.with_parent(closure_def, |this| this.visit_block(body));
                        }
                        return smallvec![i];
                    }
                    _ => {}
                }
                mut_visit::noop_flat_map_item(i, this)
            })
        })
    }

    fn visit_use_tree(&mut self, use_tree: &mut UseTree) {
        let UseTree { prefix, kind, span } = use_tree;
        self.visit_path(prefix);
        match kind {
            UseTreeKind::Simple(None) => {}
            UseTreeKind::Simple(Some(rename)) => self.visit_ident(rename),
            UseTreeKind::Nested { items, span } => {
                for (tree, id) in items {
                    self.visit_id(id);
                    // HIR lowers use trees as a flat stream of `ItemKind::Use`.
                    // This means all the def-ids must be parented to the module,
                    // and not to `self.parent_def` which is the topmost `use` item.
                    self.resolver.create_def(
                        self.resolver.tcx.local_parent(self.parent_def),
                        *id,
                        kw::Empty,
                        DefKind::Use,
                        self.expansion.to_expn_id(),
                        span.with_parent(None),
                    );
                    self.visit_use_tree(tree);
                }
                self.visit_span(span);
            }
            UseTreeKind::Glob => {}
        }
        self.visit_span(span);
    }

    fn flat_map_foreign_item(&mut self, fi: P<ForeignItem>) -> SmallVec<[P<ForeignItem>; 1]> {
        let def_kind = match fi.kind {
            ForeignItemKind::Static(box StaticItem { ty: _, mutability, expr: _, safety }) => {
                let safety = match safety {
                    ast::Safety::Unsafe(_) | ast::Safety::Default => hir::Safety::Unsafe,
                    ast::Safety::Safe(_) => hir::Safety::Safe,
                };

                DefKind::Static { safety, mutability, nested: false }
            }
            ForeignItemKind::Fn(_) => DefKind::Fn,
            ForeignItemKind::TyAlias(_) => DefKind::ForeignTy,
            ForeignItemKind::MacCall(_) => {
                self.visit_macro_invoc(fi.id);
                return smallvec![fi];
            }
        };

        let def = self.create_def(fi.id, fi.ident.name, def_kind, fi.span);

        self.with_parent(def, |this| mut_visit::noop_flat_map_item(fi, this))
    }

    fn flat_map_variant(&mut self, v: Variant) -> SmallVec<[Variant; 1]> {
        if v.is_placeholder {
            self.visit_macro_invoc(v.id);
            return smallvec![v];
        }
        let def = self.create_def(v.id, v.ident.name, DefKind::Variant, v.span);
        self.with_parent(def, |this| {
            if let Some((ctor_kind, ctor_node_id)) = CtorKind::from_ast(&v.data) {
                this.create_def(
                    ctor_node_id,
                    kw::Empty,
                    DefKind::Ctor(CtorOf::Variant, ctor_kind),
                    v.span,
                );
            }
            mut_visit::noop_flat_map_variant(v, this)
        })
    }

    fn visit_variant_data(&mut self, data: &mut VariantData) {
        // The assumption here is that non-`cfg` macro expansion cannot change field indices.
        // It currently holds because only inert attributes are accepted on fields,
        // and every such attribute expands into a single field after it's resolved.
        let fields = match data {
            VariantData::Struct { fields, recovered: _ } => fields,
            VariantData::Tuple(fields, id) => {
                self.visit_id(id);
                fields
            }
            VariantData::Unit(id) => {
                self.visit_id(id);
                return;
            }
        };
        let mut index = 0;
        fields.flat_map_in_place(|field| {
            let field = self.collect_field(field, Some(index));
            index = index + 1;
            field
        })
    }

    fn flat_map_generic_param(&mut self, param: GenericParam) -> SmallVec<[GenericParam; 1]> {
        if param.is_placeholder {
            self.visit_macro_invoc(param.id);
            return smallvec![param];
        }
        let def_kind = match param.kind {
            GenericParamKind::Lifetime { .. } => DefKind::LifetimeParam,
            GenericParamKind::Type { .. } => DefKind::TyParam,
            GenericParamKind::Const { .. } => DefKind::ConstParam,
        };
        self.create_def(param.id, param.ident.name, def_kind, param.ident.span);

        // impl-Trait can happen inside generic parameters, like
        // ```
        // fn foo<U: Iterator<Item = impl Clone>>() {}
        // ```
        //
        // In that case, the impl-trait is lowered as an additional generic parameter.
        self.with_impl_trait(ImplTraitContext::Universal, |this| {
            mut_visit::noop_flat_map_generic_param(param, this)
        })
    }

    fn flat_map_trait_item(&mut self, mut i: P<AssocItem>) -> SmallVec<[P<AssocItem>; 1]> {
        let def_kind = match &i.kind {
            AssocItemKind::Fn(..) | AssocItemKind::Delegation(..) => DefKind::AssocFn,
            AssocItemKind::Const(..) => DefKind::AssocConst,
            AssocItemKind::Type(..) => DefKind::AssocTy,
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                self.visit_macro_invoc(i.id);
                return smallvec![i];
            }
        };

        let span = i.span;
        let def = self.create_def(i.id, i.ident.name, def_kind, span);
        self.with_parent(def, |this| {
            let item = &mut *i;
            if let AssocItemKind::Fn(box Fn { defaultness, generics, sig, body }) = &mut item.kind
                && let Some(coroutine_kind) = sig.header.coroutine_kind
            {
                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                mut_visit::visit_attrs(&mut item.attrs, this);
                this.visit_vis(&mut item.vis);
                this.visit_ident(&mut item.ident);
                this.visit_span(&mut item.span);
                mut_visit::visit_defaultness(defaultness, this);
                this.visit_generics(generics);
                mut_visit::visit_fn_sig(sig, this);
                // If this async fn has no body (i.e. it's an async fn signature in a trait)
                // then the closure_def will never be used, and we should avoid generating a
                // def-id for it.
                if let Some(body) = body {
                    let closure_def = this.create_def(
                        coroutine_kind.closure_id(),
                        kw::Empty,
                        DefKind::Closure,
                        span,
                    );
                    this.with_parent(closure_def, |this| this.visit_block(body));
                }
                return smallvec![i];
            }
            mut_visit::noop_flat_map_item(i, this)
        })
    }

    fn flat_map_impl_item(&mut self, mut i: P<AssocItem>) -> SmallVec<[P<AssocItem>; 1]> {
        let def_kind = match &i.kind {
            AssocItemKind::Fn(..) | AssocItemKind::Delegation(..) => DefKind::AssocFn,
            AssocItemKind::Const(..) => DefKind::AssocConst,
            AssocItemKind::Type(..) => DefKind::AssocTy,
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                self.visit_macro_invoc(i.id);
                return smallvec![i];
            }
        };

        let span = i.span;
        let def = self.create_def(i.id, i.ident.name, def_kind, span);
        self.with_parent(def, |this| {
            let item = &mut *i;
            if let AssocItemKind::Fn(box Fn { defaultness, generics, sig, body }) = &mut item.kind
                && let Some(coroutine_kind) = sig.header.coroutine_kind
            {
                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                mut_visit::visit_attrs(&mut item.attrs, this);
                this.visit_vis(&mut item.vis);
                this.visit_ident(&mut item.ident);
                this.visit_span(&mut item.span);
                mut_visit::visit_defaultness(defaultness, this);
                this.visit_generics(generics);
                mut_visit::visit_fn_sig(sig, this);
                // If this async fn has no body (i.e. it's an async fn signature in a trait)
                // then the closure_def will never be used, and we should avoid generating a
                // def-id for it.
                if let Some(body) = body {
                    let closure_def = this.create_def(
                        coroutine_kind.closure_id(),
                        kw::Empty,
                        DefKind::Closure,
                        span,
                    );
                    this.with_parent(closure_def, |this| this.visit_block(body));
                }
                return smallvec![i];
            }
            mut_visit::noop_flat_map_item(i, this)
        })
    }

    fn visit_pat(&mut self, pat: &mut P<Pat>) {
        if let PatKind::MacCall(..) = pat.kind {
            return self.visit_macro_invoc(pat.id);
        }
        mut_visit::noop_visit_pat(pat, self)
    }

    fn visit_anon_const(&mut self, constant: &mut AnonConst) {
        let def = self.create_def(constant.id, kw::Empty, DefKind::AnonConst, constant.value.span);
        self.with_parent(def, |this| mut_visit::noop_visit_anon_const(constant, this))
    }

    fn visit_expr(&mut self, expr: &mut P<Expr>) {
        let expr = &mut **expr;
        let parent_def = match expr.kind {
            ExprKind::MacCall(..) => {
                return self.visit_macro_invoc(expr.id);
            }
            ExprKind::Closure(ref mut closure) => {
                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let closure_def = self.create_def(expr.id, kw::Empty, DefKind::Closure, expr.span);
                match closure.coroutine_kind {
                    Some(coroutine_kind) => {
                        self.with_parent(closure_def, |this| {
                            let coroutine_def = this.create_def(
                                coroutine_kind.closure_id(),
                                kw::Empty,
                                DefKind::Closure,
                                expr.span,
                            );
                            this.with_parent(coroutine_def, |this| {
                                mut_visit::noop_visit_expr(expr, this)
                            });
                        });
                        return;
                    }
                    None => closure_def,
                }
            }
            ExprKind::Gen(_, _, _, _) => {
                self.create_def(expr.id, kw::Empty, DefKind::Closure, expr.span)
            }
            ExprKind::ConstBlock(ref mut constant) => {
                mut_visit::visit_attrs(&mut expr.attrs, self);
                self.visit_span(&mut expr.span);
                let def = self.create_def(
                    constant.id,
                    kw::Empty,
                    DefKind::InlineConst,
                    constant.value.span,
                );
                self.with_parent(def, |this| mut_visit::noop_visit_anon_const(constant, this));
                return;
            }
            _ => self.parent_def,
        };

        self.with_parent(parent_def, |this| mut_visit::noop_visit_expr(expr, this))
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_ty(&mut self, ty: &mut P<Ty>) {
        match &ty.kind {
            TyKind::MacCall(..) => return self.visit_macro_invoc(ty.id),
            // Anonymous structs or unions are visited later after defined.
            TyKind::AnonStruct(..) | TyKind::AnonUnion(..) => {}
            _ => mut_visit::noop_visit_ty(ty, self),
        }
    }

    fn flat_map_stmt(&mut self, stmt: Stmt) -> SmallVec<[Stmt; 1]> {
        if let StmtKind::MacCall(..) = stmt.kind {
            self.visit_macro_invoc(stmt.id);
            return smallvec![stmt];
        }
        mut_visit::noop_flat_map_stmt(stmt, self)
    }

    fn flat_map_arm(&mut self, arm: Arm) -> SmallVec<[Arm; 1]> {
        if arm.is_placeholder {
            self.visit_macro_invoc(arm.id);
            return smallvec![arm];
        }
        mut_visit::noop_flat_map_arm(arm, self)
    }

    fn flat_map_expr_field(&mut self, f: ExprField) -> SmallVec<[ExprField; 1]> {
        if f.is_placeholder {
            self.visit_macro_invoc(f.id);
            return smallvec![f];
        }
        mut_visit::noop_flat_map_expr_field(f, self)
    }

    fn flat_map_pat_field(&mut self, fp: PatField) -> SmallVec<[PatField; 1]> {
        if fp.is_placeholder {
            self.visit_macro_invoc(fp.id);
            return smallvec![fp];
        }
        mut_visit::noop_flat_map_pat_field(fp, self)
    }

    fn flat_map_param(&mut self, p: Param) -> SmallVec<[Param; 1]> {
        if p.is_placeholder {
            self.visit_macro_invoc(p.id);
            return smallvec![p];
        }
        self.with_impl_trait(ImplTraitContext::Universal, |this| {
            mut_visit::noop_flat_map_param(p, this)
        })
    }

    // This method is called only when we are visiting an individual field
    // after expanding an attribute on it.
    fn flat_map_field_def(&mut self, field: FieldDef) -> SmallVec<[FieldDef; 1]> {
        self.collect_field(field, None)
    }

    fn visit_crate(&mut self, krate: &mut Crate) {
        if krate.is_placeholder {
            return self.visit_macro_invoc(krate.id);
        }
        mut_visit::noop_visit_crate(krate, self)
    }
}
