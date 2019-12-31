use log::debug;
use rustc::hir::def_id::DefIndex;
use rustc::hir::map::definitions::*;
use rustc_expand::expand::AstFragment;
use syntax::ast::*;
use syntax::symbol::{kw, sym};
use syntax::token::{self, Token};
use syntax::visit;
use syntax_pos::hygiene::ExpnId;
use syntax_pos::Span;

crate fn collect_definitions(
    definitions: &mut Definitions,
    fragment: &AstFragment,
    expansion: ExpnId,
) {
    let parent_def = definitions.invocation_parent(expansion);
    fragment.visit_with(&mut DefCollector { definitions, parent_def, expansion });
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a> {
    definitions: &'a mut Definitions,
    parent_def: DefIndex,
    expansion: ExpnId,
}

impl<'a> DefCollector<'a> {
    fn create_def(&mut self, node_id: NodeId, data: DefPathData, span: Span) -> DefIndex {
        let parent_def = self.parent_def;
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.definitions.create_def_with_parent(parent_def, node_id, data, self.expansion, span)
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: DefIndex, f: F) {
        let orig_parent_def = std::mem::replace(&mut self.parent_def, parent_def);
        f(self);
        self.parent_def = orig_parent_def;
    }

    fn visit_async_fn(
        &mut self,
        id: NodeId,
        name: Name,
        span: Span,
        header: &FnHeader,
        generics: &'a Generics,
        decl: &'a FnDecl,
        body: Option<&'a Block>,
    ) {
        let (closure_id, return_impl_trait_id) = match header.asyncness.node {
            IsAsync::Async { closure_id, return_impl_trait_id } => {
                (closure_id, return_impl_trait_id)
            }
            _ => unreachable!(),
        };

        // For async functions, we need to create their inner defs inside of a
        // closure to match their desugared representation.
        let fn_def_data = DefPathData::ValueNs(name);
        let fn_def = self.create_def(id, fn_def_data, span);
        return self.with_parent(fn_def, |this| {
            this.create_def(return_impl_trait_id, DefPathData::ImplTrait, span);

            visit::walk_generics(this, generics);
            visit::walk_fn_decl(this, decl);

            let closure_def = this.create_def(closure_id, DefPathData::ClosureExpr, span);
            this.with_parent(closure_def, |this| {
                if let Some(body) = body {
                    visit::walk_block(this, body);
                }
            })
        });
    }

    fn collect_field(&mut self, field: &'a StructField, index: Option<usize>) {
        let index = |this: &Self| {
            index.unwrap_or_else(|| {
                let node_id = NodeId::placeholder_from_expn_id(this.expansion);
                this.definitions.placeholder_field_index(node_id)
            })
        };

        if field.is_placeholder {
            self.definitions.set_placeholder_field_index(field.id, index(self));
            self.visit_macro_invoc(field.id);
        } else {
            let name = field.ident.map_or_else(|| sym::integer(index(self)), |ident| ident.name);
            let def = self.create_def(field.id, DefPathData::ValueNs(name), field.span);
            self.with_parent(def, |this| visit::walk_struct_field(this, field));
        }
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        self.definitions.set_invocation_parent(id.placeholder_to_expn_id(), self.parent_def);
    }
}

impl<'a> visit::Visitor<'a> for DefCollector<'a> {
    fn visit_item(&mut self, i: &'a Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into, the better
        let def_data = match &i.kind {
            ItemKind::Impl(..) => DefPathData::Impl,
            ItemKind::Mod(..) if i.ident.name == kw::Invalid => {
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
            ItemKind::Fn(sig, generics, body) if sig.header.asyncness.node.is_async() => {
                return self.visit_async_fn(
                    i.id,
                    i.ident.name,
                    i.span,
                    &sig.header,
                    generics,
                    &sig.decl,
                    Some(body),
                );
            }
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) => {
                DefPathData::ValueNs(i.ident.name)
            }
            ItemKind::MacroDef(..) => DefPathData::MacroNs(i.ident.name),
            ItemKind::Mac(..) => return self.visit_macro_invoc(i.id),
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

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, DefPathData::Misc, use_tree.span);
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, foreign_item: &'a ForeignItem) {
        if let ForeignItemKind::Macro(_) = foreign_item.kind {
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

    fn visit_trait_item(&mut self, ti: &'a AssocItem) {
        let def_data = match ti.kind {
            AssocItemKind::Fn(..) | AssocItemKind::Const(..) => DefPathData::ValueNs(ti.ident.name),
            AssocItemKind::TyAlias(..) => DefPathData::TypeNs(ti.ident.name),
            AssocItemKind::Macro(..) => return self.visit_macro_invoc(ti.id),
        };

        let def = self.create_def(ti.id, def_data, ti.span);
        self.with_parent(def, |this| visit::walk_trait_item(this, ti));
    }

    fn visit_impl_item(&mut self, ii: &'a AssocItem) {
        let def_data = match ii.kind {
            AssocItemKind::Fn(FnSig { ref header, ref decl }, ref body)
                if header.asyncness.node.is_async() =>
            {
                return self.visit_async_fn(
                    ii.id,
                    ii.ident.name,
                    ii.span,
                    header,
                    &ii.generics,
                    decl,
                    body.as_deref(),
                );
            }
            AssocItemKind::Fn(..) | AssocItemKind::Const(..) => DefPathData::ValueNs(ii.ident.name),
            AssocItemKind::TyAlias(..) => DefPathData::TypeNs(ii.ident.name),
            AssocItemKind::Macro(..) => return self.visit_macro_invoc(ii.id),
        };

        let def = self.create_def(ii.id, def_data, ii.span);
        self.with_parent(def, |this| visit::walk_impl_item(this, ii));
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.kind {
            PatKind::Mac(..) => return self.visit_macro_invoc(pat.id),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_anon_const(&mut self, constant: &'a AnonConst) {
        let def = self.create_def(constant.id, DefPathData::AnonConst, constant.value.span);
        self.with_parent(def, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = match expr.kind {
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(_, asyncness, ..) => {
                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let closure_def = self.create_def(expr.id, DefPathData::ClosureExpr, expr.span);
                match asyncness {
                    IsAsync::Async { closure_id, .. } => {
                        self.create_def(closure_id, DefPathData::ClosureExpr, expr.span)
                    }
                    IsAsync::NotAsync => closure_def,
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
            TyKind::Mac(..) => return self.visit_macro_invoc(ty.id),
            TyKind::ImplTrait(node_id, _) => {
                self.create_def(node_id, DefPathData::ImplTrait, ty.span);
            }
            _ => {}
        }
        visit::walk_ty(self, ty);
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt.kind {
            StmtKind::Mac(..) => self.visit_macro_invoc(stmt.id),
            _ => visit::walk_stmt(self, stmt),
        }
    }

    fn visit_token(&mut self, t: Token) {
        if let token::Interpolated(nt) = t.kind {
            if let token::NtExpr(ref expr) = *nt {
                if let ExprKind::Mac(..) = expr.kind {
                    self.visit_macro_invoc(expr.id);
                }
            }
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
