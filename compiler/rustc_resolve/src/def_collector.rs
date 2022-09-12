use crate::{ImplTraitContext, Resolver};
use rustc_ast::visit::{self, FnKind};
use rustc_ast::walk_list;
use rustc_ast::*;
use rustc_expand::expand::AstFragment;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::definitions::*;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::symbol::sym;
use rustc_span::Span;

pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_>,
    fragment: &AstFragment,
    expansion: LocalExpnId,
) {
    let (parent_def, impl_trait_context) = resolver.invocation_parents[&expansion];
    fragment.visit_with(&mut DefCollector { resolver, parent_def, expansion, impl_trait_context });
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a, 'b> {
    resolver: &'a mut Resolver<'b>,
    parent_def: LocalDefId,
    impl_trait_context: ImplTraitContext,
    expansion: LocalExpnId,
}

impl<'a, 'b> DefCollector<'a, 'b> {
    fn create_def(&mut self, node_id: NodeId, data: DefPathData, span: Span) -> LocalDefId {
        let parent_def = self.parent_def;
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.resolver.create_def(
            parent_def,
            node_id,
            data,
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
            let def = self.create_def(field.id, DefPathData::ValueNs(name), field.span);
            self.with_parent(def, |this| visit::walk_field_def(this, field));
        }
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        let id = id.placeholder_to_expn_id();
        let old_parent =
            self.resolver.invocation_parents.insert(id, (self.parent_def, self.impl_trait_context));
        assert!(old_parent.is_none(), "parent `LocalDefId` is reset for an invocation");
    }
}

impl<'a, 'b> visit::Visitor<'a> for DefCollector<'a, 'b> {
    fn visit_item(&mut self, i: &'a Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into, the better
        let def_data = match &i.kind {
            ItemKind::Impl { .. } => DefPathData::Impl,
            ItemKind::ForeignMod(..) => DefPathData::ForeignMod,
            ItemKind::Mod(..)
            | ItemKind::Trait(..)
            | ItemKind::TraitAlias(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::TyAlias(..) => DefPathData::TypeNs(i.ident.name),
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) => {
                DefPathData::ValueNs(i.ident.name)
            }
            ItemKind::MacroDef(..) => DefPathData::MacroNs(i.ident.name),
            ItemKind::MacCall(..) => {
                visit::walk_item(self, i);
                return self.visit_macro_invoc(i.id);
            }
            ItemKind::GlobalAsm(..) => DefPathData::GlobalAsm,
            ItemKind::Use(..) => {
                return visit::walk_item(self, i);
            }
        };
        let def = self.create_def(i.id, def_data, i.span);

        self.with_parent(def, |this| {
            this.with_impl_trait(ImplTraitContext::Existential, |this| {
                match i.kind {
                    ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                        // If this is a unit or tuple-like struct, register the constructor.
                        if let Some(ctor_hir_id) = struct_def.ctor_id() {
                            this.create_def(ctor_hir_id, DefPathData::Ctor, i.span);
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
            if let Async::Yes { closure_id, return_impl_trait_id, .. } = sig.header.asyncness {
                self.visit_generics(generics);

                let return_impl_trait_id =
                    self.create_def(return_impl_trait_id, DefPathData::ImplTrait, span);

                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                self.visit_fn_header(&sig.header);
                for param in &sig.decl.inputs {
                    self.visit_param(param);
                }
                self.with_parent(return_impl_trait_id, |this| {
                    this.visit_fn_ret_ty(&sig.decl.output)
                });
                let closure_def = self.create_def(closure_id, DefPathData::ClosureExpr, span);
                self.with_parent(closure_def, |this| walk_list!(this, visit_block, body));
                return;
            }
        }

        visit::walk_fn(self, fn_kind, span);
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, DefPathData::Use, use_tree.span);
        match use_tree.kind {
            UseTreeKind::Simple(_, id1, id2) => {
                self.create_def(id1, DefPathData::Use, use_tree.prefix.span);
                self.create_def(id2, DefPathData::Use, use_tree.prefix.span);
            }
            UseTreeKind::Glob => (),
            UseTreeKind::Nested(..) => {}
        }
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, foreign_item: &'a ForeignItem) {
        if let ForeignItemKind::MacCall(_) = foreign_item.kind {
            return self.visit_macro_invoc(foreign_item.id);
        }

        let def = self.create_def(
            foreign_item.id,
            DefPathData::ValueNs(foreign_item.ident.name),
            foreign_item.span,
        );

        self.with_parent(def, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_variant(&mut self, v: &'a Variant) {
        if v.is_placeholder {
            return self.visit_macro_invoc(v.id);
        }
        let def = self.create_def(v.id, DefPathData::TypeNs(v.ident.name), v.span);
        self.with_parent(def, |this| {
            if let Some(ctor_hir_id) = v.data.ctor_id() {
                this.create_def(ctor_hir_id, DefPathData::Ctor, v.span);
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
        let name = param.ident.name;
        let def_path_data = match param.kind {
            GenericParamKind::Lifetime { .. } => DefPathData::LifetimeNs(name),
            GenericParamKind::Type { .. } => DefPathData::TypeNs(name),
            GenericParamKind::Const { .. } => DefPathData::ValueNs(name),
        };
        self.create_def(param.id, def_path_data, param.ident.span);

        // impl-Trait can happen inside generic parameters, like
        // ```
        // fn foo<U: Iterator<Item = impl Clone>>() {}
        // ```
        //
        // In that case, the impl-trait is lowered as an additional generic parameter.
        self.with_impl_trait(ImplTraitContext::Universal(self.parent_def), |this| {
            visit::walk_generic_param(this, param)
        });
    }

    fn visit_assoc_item(&mut self, i: &'a AssocItem, ctxt: visit::AssocCtxt) {
        let def_data = match &i.kind {
            AssocItemKind::Fn(..) | AssocItemKind::Const(..) => DefPathData::ValueNs(i.ident.name),
            AssocItemKind::TyAlias(..) => DefPathData::TypeNs(i.ident.name),
            AssocItemKind::MacCall(..) => return self.visit_macro_invoc(i.id),
        };

        let def = self.create_def(i.id, def_data, i.span);
        self.with_parent(def, |this| visit::walk_assoc_item(this, i, ctxt));
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.kind {
            PatKind::MacCall(..) => self.visit_macro_invoc(pat.id),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_anon_const(&mut self, constant: &'a AnonConst) {
        let def = self.create_def(constant.id, DefPathData::AnonConst, constant.value.span);
        self.with_parent(def, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = match expr.kind {
            ExprKind::MacCall(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(_, _, asyncness, ..) => {
                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let closure_def = self.create_def(expr.id, DefPathData::ClosureExpr, expr.span);
                match asyncness {
                    Async::Yes { closure_id, .. } => {
                        self.create_def(closure_id, DefPathData::ClosureExpr, expr.span)
                    }
                    Async::No => closure_def,
                }
            }
            ExprKind::Async(_, async_id, _) => {
                self.create_def(async_id, DefPathData::ClosureExpr, expr.span)
            }
            _ => self.parent_def,
        };

        self.with_parent(parent_def, |this| visit::walk_expr(this, expr));
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.kind {
            TyKind::MacCall(..) => self.visit_macro_invoc(ty.id),
            TyKind::ImplTrait(node_id, _) => {
                let parent_def = match self.impl_trait_context {
                    ImplTraitContext::Universal(item_def) => self.resolver.create_def(
                        item_def,
                        node_id,
                        DefPathData::ImplTrait,
                        self.expansion.to_expn_id(),
                        ty.span,
                    ),
                    ImplTraitContext::Existential => {
                        self.create_def(node_id, DefPathData::ImplTrait, ty.span)
                    }
                };
                self.with_parent(parent_def, |this| visit::walk_ty(this, ty))
            }
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
            self.with_impl_trait(ImplTraitContext::Universal(self.parent_def), |this| {
                visit::walk_param(this, p)
            })
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
