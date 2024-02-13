use crate::{ImplTraitContext, Resolver};
use rustc_ast::visit::{self, FnKind};
use rustc_ast::*;
use rustc_expand::expand::AstFragment;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_, '_>,
    fragment: &AstFragment,
    expansion: LocalExpnId,
) {
    let (parent_def, impl_trait_context) = resolver.invocation_parents[&expansion];
    fragment.visit_with(&mut DefCollector { resolver, parent_def, expansion, impl_trait_context });
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
        self.resolver.create_def(
            parent_def,
            node_id,
            name,
            def_kind,
            self.expansion.to_expn_id(),
            span.with_parent(None),
        )
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: LocalDefId, f: F) {
        let orig_parent_def = std::mem::replace(&mut self.parent_def, parent_def);
        f(self);
        self.parent_def = orig_parent_def;
    }

    fn with_impl_trait<F: FnOnce(&mut Self)>(
        &mut self,
        impl_trait_context: ImplTraitContext,
        f: F,
    ) {
        let orig_itc = std::mem::replace(&mut self.impl_trait_context, impl_trait_context);
        f(self);
        self.impl_trait_context = orig_itc;
    }

    fn collect_field(&mut self, field: &'a FieldDef, index: Option<usize>) {
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
        } else {
            let name = field.ident.map_or_else(|| sym::integer(index(self)), |ident| ident.name);
            let def = self.create_def(field.id, name, DefKind::Field, field.span);
            self.with_parent(def, |this| visit::walk_field_def(this, field));
            self.visit_anon_adt(&field.ty);
        }
    }

    fn visit_anon_adt(&mut self, ty: &'a Ty) {
        let def_kind = match &ty.kind {
            TyKind::AnonStruct(..) => DefKind::Struct,
            TyKind::AnonUnion(..) => DefKind::Union,
            _ => return,
        };
        match &ty.kind {
            TyKind::AnonStruct(node_id, _) | TyKind::AnonUnion(node_id, _) => {
                let def_id = self.create_def(*node_id, kw::Empty, def_kind, ty.span);
                self.with_parent(def_id, |this| visit::walk_ty(this, ty));
            }
            _ => {}
        }
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        let id = id.placeholder_to_expn_id();
        let old_parent =
            self.resolver.invocation_parents.insert(id, (self.parent_def, self.impl_trait_context));
        assert!(old_parent.is_none(), "parent `LocalDefId` is reset for an invocation");
    }
}

impl<'a, 'b, 'tcx> visit::Visitor<'a> for DefCollector<'a, 'b, 'tcx> {
    fn visit_item(&mut self, i: &'a Item) {
        debug!("visit_item: {:?}", i);

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
            ItemKind::Static(s) => DefKind::Static(s.mutability),
            ItemKind::Const(..) => DefKind::Const,
            ItemKind::Fn(..) | ItemKind::Delegation(..) => DefKind::Fn,
            ItemKind::MacroDef(..) => {
                let macro_data = self.resolver.compile_macro(i, self.resolver.tcx.sess.edition());
                let macro_kind = macro_data.ext.macro_kind();
                opt_macro_data = Some(macro_data);
                DefKind::Macro(macro_kind)
            }
            ItemKind::MacCall(..) => {
                visit::walk_item(self, i);
                return self.visit_macro_invoc(i.id);
            }
            ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
            ItemKind::Use(..) => {
                return visit::walk_item(self, i);
            }
        };
        let def_id = self.create_def(i.id, i.ident.name, def_kind, i.span);

        if let Some(macro_data) = opt_macro_data {
            self.resolver.macro_map.insert(def_id.to_def_id(), macro_data);
        }

        self.with_parent(def_id, |this| {
            this.with_impl_trait(ImplTraitContext::Existential, |this| {
                match i.kind {
                    ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                        // If this is a unit or tuple-like struct, register the constructor.
                        if let Some((ctor_kind, ctor_node_id)) = CtorKind::from_ast(struct_def) {
                            this.create_def(
                                ctor_node_id,
                                kw::Empty,
                                DefKind::Ctor(CtorOf::Struct, ctor_kind),
                                i.span,
                            );
                        }
                    }
                    _ => {}
                }
                visit::walk_item(this, i);
            })
        });
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, span: Span, _: NodeId) {
        if let FnKind::Fn(_, _, sig, _, generics, body) = fn_kind {
            match sig.header.coroutine_kind {
                Some(coroutine_kind) => {
                    self.visit_generics(generics);

                    // For async functions, we need to create their inner defs inside of a
                    // closure to match their desugared representation. Besides that,
                    // we must mirror everything that `visit::walk_fn` below does.
                    self.visit_fn_header(&sig.header);
                    for param in &sig.decl.inputs {
                        self.visit_param(param);
                    }
                    self.visit_fn_ret_ty(&sig.decl.output);
                    // If this async fn has no body (i.e. it's an async fn signature in a trait)
                    // then the closure_def will never be used, and we should avoid generating a
                    // def-id for it.
                    if let Some(body) = body {
                        let closure_def = self.create_def(
                            coroutine_kind.closure_id(),
                            kw::Empty,
                            DefKind::Closure,
                            span,
                        );
                        self.with_parent(closure_def, |this| this.visit_block(body));
                    }
                    return;
                }
                None => {}
            }
        }

        visit::walk_fn(self, fn_kind);
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, kw::Empty, DefKind::Use, use_tree.span);
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        let def_kind = match fi.kind {
            ForeignItemKind::Static(_, mt, _) => DefKind::Static(mt),
            ForeignItemKind::Fn(_) => DefKind::Fn,
            ForeignItemKind::TyAlias(_) => DefKind::ForeignTy,
            ForeignItemKind::MacCall(_) => return self.visit_macro_invoc(fi.id),
        };

        let def = self.create_def(fi.id, fi.ident.name, def_kind, fi.span);

        self.with_parent(def, |this| visit::walk_foreign_item(this, fi));
    }

    fn visit_variant(&mut self, v: &'a Variant) {
        if v.is_placeholder {
            return self.visit_macro_invoc(v.id);
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
            visit::walk_variant(this, v)
        });
    }

    fn visit_variant_data(&mut self, data: &'a VariantData) {
        // The assumption here is that non-`cfg` macro expansion cannot change field indices.
        // It currently holds because only inert attributes are accepted on fields,
        // and every such attribute expands into a single field after it's resolved.
        for (index, field) in data.fields().iter().enumerate() {
            self.collect_field(field, Some(index));
        }
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        if param.is_placeholder {
            self.visit_macro_invoc(param.id);
            return;
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
            visit::walk_generic_param(this, param)
        });
    }

    fn visit_assoc_item(&mut self, i: &'a AssocItem, ctxt: visit::AssocCtxt) {
        let def_kind = match &i.kind {
            AssocItemKind::Fn(..) | AssocItemKind::Delegation(..) => DefKind::AssocFn,
            AssocItemKind::Const(..) => DefKind::AssocConst,
            AssocItemKind::Type(..) => DefKind::AssocTy,
            AssocItemKind::MacCall(..) => return self.visit_macro_invoc(i.id),
        };

        let def = self.create_def(i.id, i.ident.name, def_kind, i.span);
        self.with_parent(def, |this| visit::walk_assoc_item(this, i, ctxt));
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.kind {
            PatKind::MacCall(..) => self.visit_macro_invoc(pat.id),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_anon_const(&mut self, constant: &'a AnonConst) {
        let def = self.create_def(constant.id, kw::Empty, DefKind::AnonConst, constant.value.span);
        self.with_parent(def, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = match expr.kind {
            ExprKind::MacCall(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(ref closure) => {
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
                            this.with_parent(coroutine_def, |this| visit::walk_expr(this, expr));
                        });
                        return;
                    }
                    None => closure_def,
                }
            }
            ExprKind::Gen(_, _, _) => {
                self.create_def(expr.id, kw::Empty, DefKind::Closure, expr.span)
            }
            ExprKind::ConstBlock(ref constant) => {
                let def = self.create_def(
                    constant.id,
                    kw::Empty,
                    DefKind::InlineConst,
                    constant.value.span,
                );
                self.with_parent(def, |this| visit::walk_anon_const(this, constant));
                return;
            }
            _ => self.parent_def,
        };

        self.with_parent(parent_def, |this| visit::walk_expr(this, expr));
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match &ty.kind {
            TyKind::MacCall(..) => self.visit_macro_invoc(ty.id),
            // Anonymous structs or unions are visited later after defined.
            TyKind::AnonStruct(..) | TyKind::AnonUnion(..) => {}
            _ => visit::walk_ty(self, ty),
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt.kind {
            StmtKind::MacCall(..) => self.visit_macro_invoc(stmt.id),
            _ => visit::walk_stmt(self, stmt),
        }
    }

    fn visit_arm(&mut self, arm: &'a Arm) {
        if arm.is_placeholder { self.visit_macro_invoc(arm.id) } else { visit::walk_arm(self, arm) }
    }

    fn visit_expr_field(&mut self, f: &'a ExprField) {
        if f.is_placeholder {
            self.visit_macro_invoc(f.id)
        } else {
            visit::walk_expr_field(self, f)
        }
    }

    fn visit_pat_field(&mut self, fp: &'a PatField) {
        if fp.is_placeholder {
            self.visit_macro_invoc(fp.id)
        } else {
            visit::walk_pat_field(self, fp)
        }
    }

    fn visit_param(&mut self, p: &'a Param) {
        if p.is_placeholder {
            self.visit_macro_invoc(p.id)
        } else {
            self.with_impl_trait(ImplTraitContext::Universal, |this| visit::walk_param(this, p))
        }
    }

    // This method is called only when we are visiting an individual field
    // after expanding an attribute on it.
    fn visit_field_def(&mut self, field: &'a FieldDef) {
        self.collect_field(field, None);
    }

    fn visit_crate(&mut self, krate: &'a Crate) {
        if krate.is_placeholder {
            self.visit_macro_invoc(krate.id)
        } else {
            visit::walk_crate(self, krate)
        }
    }
}
