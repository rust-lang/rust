use std::mem;

use rustc_ast::visit::FnKind;
use rustc_ast::*;
use rustc_ast_pretty::pprust;
use rustc_expand::expand::AstFragment;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use tracing::debug;

use crate::{ImplTraitContext, InvocationParent, PendingAnonConstInfo, Resolver};

pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_, '_>,
    fragment: &AstFragment,
    expansion: LocalExpnId,
) {
    let InvocationParent { parent_def, pending_anon_const_info, impl_trait_context, in_attr } =
        resolver.invocation_parents[&expansion];
    let mut visitor = DefCollector {
        resolver,
        parent_def,
        pending_anon_const_info,
        expansion,
        impl_trait_context,
        in_attr,
    };
    fragment.visit_with(&mut visitor);
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a, 'ra, 'tcx> {
    resolver: &'a mut Resolver<'ra, 'tcx>,
    parent_def: LocalDefId,
    /// If we have an anon const that consists of a macro invocation, e.g. `Foo<{ m!() }>`,
    /// we need to wait until we know what the macro expands to before we create the def for
    /// the anon const. That's because we lower some anon consts into `hir::ConstArgKind::Path`,
    /// which don't have defs.
    ///
    /// See `Self::visit_anon_const()`.
    pending_anon_const_info: Option<PendingAnonConstInfo>,
    impl_trait_context: ImplTraitContext,
    in_attr: bool,
    expansion: LocalExpnId,
}

impl<'a, 'ra, 'tcx> DefCollector<'a, 'ra, 'tcx> {
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

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: LocalDefId, f: F) {
        let orig_parent_def = mem::replace(&mut self.parent_def, parent_def);
        f(self);
        self.parent_def = orig_parent_def;
    }

    fn with_impl_trait<F: FnOnce(&mut Self)>(
        &mut self,
        impl_trait_context: ImplTraitContext,
        f: F,
    ) {
        let orig_itc = mem::replace(&mut self.impl_trait_context, impl_trait_context);
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
        let pending_anon_const_info = self.pending_anon_const_info.take();
        let old_parent = self.resolver.invocation_parents.insert(
            id,
            InvocationParent {
                parent_def: self.parent_def,
                pending_anon_const_info,
                impl_trait_context: self.impl_trait_context,
                in_attr: self.in_attr,
            },
        );
        assert!(old_parent.is_none(), "parent `LocalDefId` is reset for an invocation");
    }
}

impl<'a, 'ra, 'tcx> visit::Visitor<'a> for DefCollector<'a, 'ra, 'tcx> {
    fn visit_item(&mut self, i: &'a Item) {
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
                let macro_data = self.resolver.compile_macro(i, self.resolver.tcx.sess.edition());
                let macro_kind = macro_data.ext.macro_kind();
                opt_macro_data = Some(macro_data);
                DefKind::Macro(macro_kind)
            }
            ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
            ItemKind::Use(..) => return visit::walk_item(self, i),
            ItemKind::MacCall(..) | ItemKind::DelegationMac(..) => {
                return self.visit_macro_invoc(i.id);
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
        match fn_kind {
            FnKind::Fn(_ctxt, _ident, FnSig { header, decl, span: _ }, _vis, generics, body)
                if let Some(coroutine_kind) = header.coroutine_kind =>
            {
                self.visit_fn_header(header);
                self.visit_generics(generics);

                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                let FnDecl { inputs, output } = &**decl;
                for param in inputs {
                    self.visit_param(param);
                }

                let (return_id, return_span) = coroutine_kind.return_id();
                let return_def =
                    self.create_def(return_id, kw::Empty, DefKind::OpaqueTy, return_span);
                self.with_parent(return_def, |this| this.visit_fn_ret_ty(output));

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
            }
            FnKind::Closure(binder, Some(coroutine_kind), decl, body) => {
                self.visit_closure_binder(binder);
                visit::walk_fn_decl(self, decl);

                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let coroutine_def =
                    self.create_def(coroutine_kind.closure_id(), kw::Empty, DefKind::Closure, span);
                self.with_parent(coroutine_def, |this| this.visit_expr(body));
            }
            _ => visit::walk_fn(self, fn_kind),
        }
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, kw::Empty, DefKind::Use, use_tree.span);
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
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
            ForeignItemKind::MacCall(_) => return self.visit_macro_invoc(fi.id),
        };

        let def = self.create_def(fi.id, fi.ident.name, def_kind, fi.span);

        self.with_parent(def, |this| visit::walk_item(this, fi));
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
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                return self.visit_macro_invoc(i.id);
            }
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
        // HACK(min_generic_const_args): don't create defs for anon consts if we think they will
        // later be turned into ConstArgKind::Path's. because this is before resolve is done, we
        // may accidentally identify a construction of a unit struct as a param and not create a
        // def. we'll then create a def later in ast lowering in this case. the parent of nested
        // items will be messed up, but that's ok because there can't be any if we're just looking
        // for bare idents.

        if matches!(constant.value.maybe_unwrap_block().kind, ExprKind::MacCall(..)) {
            // See self.pending_anon_const_info for explanation
            self.pending_anon_const_info =
                Some(PendingAnonConstInfo { id: constant.id, span: constant.value.span });
            return visit::walk_anon_const(self, constant);
        } else if constant.value.is_potential_trivial_const_arg() {
            return visit::walk_anon_const(self, constant);
        }

        let def = self.create_def(constant.id, kw::Empty, DefKind::AnonConst, constant.value.span);
        self.with_parent(def, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        if matches!(expr.kind, ExprKind::MacCall(..)) {
            return self.visit_macro_invoc(expr.id);
        }

        let grandparent_def = if let Some(pending_anon) = self.pending_anon_const_info.take() {
            // See self.pending_anon_const_info for explanation
            if !expr.is_potential_trivial_const_arg() {
                self.create_def(pending_anon.id, kw::Empty, DefKind::AnonConst, pending_anon.span)
            } else {
                self.parent_def
            }
        } else {
            self.parent_def
        };

        self.with_parent(grandparent_def, |this| {
            let parent_def = match expr.kind {
                ExprKind::Closure(..) | ExprKind::Gen(..) => {
                    this.create_def(expr.id, kw::Empty, DefKind::Closure, expr.span)
                }
                ExprKind::ConstBlock(ref constant) => {
                    for attr in &expr.attrs {
                        visit::walk_attribute(this, attr);
                    }
                    let def = this.create_def(
                        constant.id,
                        kw::Empty,
                        DefKind::InlineConst,
                        constant.value.span,
                    );
                    this.with_parent(def, |this| visit::walk_anon_const(this, constant));
                    return;
                }
                _ => this.parent_def,
            };

            this.with_parent(parent_def, |this| visit::walk_expr(this, expr))
        })
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match &ty.kind {
            TyKind::MacCall(..) => self.visit_macro_invoc(ty.id),
            // Anonymous structs or unions are visited later after defined.
            TyKind::AnonStruct(..) | TyKind::AnonUnion(..) => {}
            TyKind::ImplTrait(id, _) => {
                // HACK: pprust breaks strings with newlines when the type
                // gets too long. We don't want these to show up in compiler
                // output or built artifacts, so replace them here...
                // Perhaps we should instead format APITs more robustly.
                let name = Symbol::intern(&pprust::ty_to_string(ty).replace('\n', " "));
                let kind = match self.impl_trait_context {
                    ImplTraitContext::Universal => DefKind::TyParam,
                    ImplTraitContext::Existential => DefKind::OpaqueTy,
                };
                let id = self.create_def(*id, name, kind, ty.span);
                match self.impl_trait_context {
                    // Do not nest APIT, as we desugar them as `impl_trait: bounds`,
                    // so the `impl_trait` node is not a parent to `bounds`.
                    ImplTraitContext::Universal => visit::walk_ty(self, ty),
                    ImplTraitContext::Existential => {
                        self.with_parent(id, |this| visit::walk_ty(this, ty))
                    }
                };
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

    fn visit_attribute(&mut self, attr: &'a Attribute) -> Self::Result {
        let orig_in_attr = mem::replace(&mut self.in_attr, true);
        visit::walk_attribute(self, attr);
        self.in_attr = orig_in_attr;
    }
}
