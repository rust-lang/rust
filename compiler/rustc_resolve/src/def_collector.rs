use crate::Resolver;
use rustc_ast::visit::{self, FnKind};
use rustc_ast::walk_list;
use rustc_ast::*;
use rustc_ast_lowering::ResolverAstLowering;
use rustc_expand::expand::AstFragment;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::definitions::*;
use rustc_span::hygiene::ExpnId;
use rustc_span::symbol::{kw, sym};
use rustc_span::Span;
use tracing::debug;

crate fn collect_definitions(
    resolver: &mut Resolver<'_>,
    fragment: &AstFragment,
    expansion: ExpnId,
) {
    let parent_def = resolver.invocation_parents[&expansion];
    fragment.visit_with(&mut DefCollector { resolver, parent_def, expansion });
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a, 'b> {
    resolver: &'a mut Resolver<'b>,
    parent_def: LocalDefId,
    expansion: ExpnId,
}

impl<'a, 'b> DefCollector<'a, 'b> {
    fn create_def(&mut self, node_id: NodeId, data: DefPathData, span: Span) -> LocalDefId {
        let parent_def = self.parent_def;
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.resolver.create_def(parent_def, node_id, data, self.expansion, span)
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: LocalDefId, f: F) {
        let orig_parent_def = std::mem::replace(&mut self.parent_def, parent_def);
        f(self);
        self.parent_def = orig_parent_def;
    }

    fn collect_field(&mut self, field: &'a StructField, index: Option<usize>) {
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
            self.with_parent(def, |this| visit::walk_struct_field(this, field));
        }
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        let old_parent =
            self.resolver.invocation_parents.insert(id.placeholder_to_expn_id(), self.parent_def);
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
            ItemKind::Mod(..) if i.ident.name == kw::Invalid => {
                // Fake crate root item from expand.
                return visit::walk_item(self, i);
            }
            ItemKind::Mod(..)
            | ItemKind::Trait(..)
            | ItemKind::TraitAlias(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::ForeignMod(..)
            | ItemKind::TyAlias(..) => DefPathData::TypeNs(i.ident.name),
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) => {
                DefPathData::ValueNs(i.ident.name)
            }
            ItemKind::MacroDef(..) => DefPathData::MacroNs(i.ident.name),
            ItemKind::MacCall(..) => return self.visit_macro_invoc(i.id),
            ItemKind::GlobalAsm(..) => DefPathData::Misc,
            ItemKind::Use(..) => {
                return visit::walk_item(self, i);
            }
        };
        let def = self.create_def(i.id, def_data, i.span);

        self.with_parent(def, |this| {
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
        });
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'a>, span: Span, _: NodeId) {
        if let FnKind::Fn(_, _, sig, _, body) = fn_kind {
            if let Async::Yes { closure_id, return_impl_trait_id, .. } = sig.header.asyncness {
                self.create_def(return_impl_trait_id, DefPathData::ImplTrait, span);

                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                self.visit_fn_header(&sig.header);
                visit::walk_fn_decl(self, &sig.decl);
                let closure_def = self.create_def(closure_id, DefPathData::ClosureExpr, span);
                self.with_parent(closure_def, |this| walk_list!(this, visit_block, body));
                return;
            }
        }

        visit::walk_fn(self, fn_kind, span);
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, DefPathData::Misc, use_tree.span);
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

        visit::walk_generic_param(self, param);
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
            ExprKind::Closure(_, asyncness, ..) => {
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
                let parent_def = self.create_def(node_id, DefPathData::ImplTrait, ty.span);
                self.with_parent(parent_def, |this| visit::walk_ty(this, ty));
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

    fn visit_field(&mut self, f: &'a Field) {
        if f.is_placeholder { self.visit_macro_invoc(f.id) } else { visit::walk_field(self, f) }
    }

    fn visit_field_pattern(&mut self, fp: &'a FieldPat) {
        if fp.is_placeholder {
            self.visit_macro_invoc(fp.id)
        } else {
            visit::walk_field_pattern(self, fp)
        }
    }

    fn visit_param(&mut self, p: &'a Param) {
        if p.is_placeholder { self.visit_macro_invoc(p.id) } else { visit::walk_param(self, p) }
    }

    // This method is called only when we are visiting an individual field
    // after expanding an attribute on it.
    fn visit_struct_field(&mut self, field: &'a StructField) {
        self.collect_field(field, None);
    }
}
