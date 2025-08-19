use std::mem;

use rustc_ast::visit::FnKind;
use rustc_ast::*;
use rustc_attr_parsing::{AttributeParser, Early, OmitDoc, ShouldEmit};
use rustc_expand::expand::AstFragment;
use rustc_hir as hir;
use rustc_hir::Target;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::span_bug;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::{Span, Symbol, sym};
use tracing::debug;

use crate::{ImplTraitContext, InvocationParent, Resolver};

pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_, '_>,
    fragment: &AstFragment,
    expansion: LocalExpnId,
) {
    let invocation_parent = resolver.invocation_parents[&expansion];
    let mut visitor = DefCollector { resolver, expansion, invocation_parent };
    fragment.visit_with(&mut visitor);
}

/// Creates `DefId`s for nodes in the AST.
struct DefCollector<'a, 'ra, 'tcx> {
    resolver: &'a mut Resolver<'ra, 'tcx>,
    invocation_parent: InvocationParent,
    expansion: LocalExpnId,
}

impl<'a, 'ra, 'tcx> DefCollector<'a, 'ra, 'tcx> {
    fn create_def(
        &mut self,
        node_id: NodeId,
        name: Option<Symbol>,
        def_kind: DefKind,
        span: Span,
    ) -> LocalDefId {
        let parent_def = self.invocation_parent.parent_def;
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
        let orig_parent_def = mem::replace(&mut self.invocation_parent.parent_def, parent_def);
        f(self);
        self.invocation_parent.parent_def = orig_parent_def;
    }

    fn with_impl_trait<F: FnOnce(&mut Self)>(
        &mut self,
        impl_trait_context: ImplTraitContext,
        f: F,
    ) {
        let orig_itc =
            mem::replace(&mut self.invocation_parent.impl_trait_context, impl_trait_context);
        f(self);
        self.invocation_parent.impl_trait_context = orig_itc;
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
            let def = self.create_def(field.id, Some(name), DefKind::Field, field.span);
            self.with_parent(def, |this| visit::walk_field_def(this, field));
        }
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        let id = id.placeholder_to_expn_id();
        let old_parent = self.resolver.invocation_parents.insert(id, self.invocation_parent);
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
            ItemKind::MacroDef(ident, def) => {
                let edition = i.span.edition();

                // FIXME(jdonszelmann) make one of these in the resolver?
                // FIXME(jdonszelmann) don't care about tools here maybe? Just parse what we can.
                // Does that prevents errors from happening? maybe
                let mut parser = AttributeParser::<'_, Early>::new(
                    &self.resolver.tcx.sess,
                    self.resolver.tcx.features(),
                    Vec::new(),
                    Early { emit_errors: ShouldEmit::Nothing },
                );
                let attrs = parser.parse_attribute_list(
                    &i.attrs,
                    i.span,
                    i.id,
                    Target::MacroDef,
                    OmitDoc::Skip,
                    std::convert::identity,
                    |_l| {
                        // FIXME(jdonszelmann): emit lints here properly
                        // NOTE that before new attribute parsing, they didn't happen either
                        // but it would be nice if we could change that.
                    },
                );

                let macro_data =
                    self.resolver.compile_macro(def, *ident, &attrs, i.span, i.id, edition);
                let macro_kinds = macro_data.ext.macro_kinds();
                opt_macro_data = Some(macro_data);
                DefKind::Macro(macro_kinds)
            }
            ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
            ItemKind::Use(use_tree) => {
                self.create_def(i.id, None, DefKind::Use, use_tree.span);
                return visit::walk_item(self, i);
            }
            ItemKind::MacCall(..) | ItemKind::DelegationMac(..) => {
                return self.visit_macro_invoc(i.id);
            }
        };
        let def_id =
            self.create_def(i.id, i.kind.ident().map(|ident| ident.name), def_kind, i.span);

        if let Some(macro_data) = opt_macro_data {
            self.resolver.new_local_macro(def_id, macro_data);
        }

        self.with_parent(def_id, |this| {
            this.with_impl_trait(ImplTraitContext::Existential, |this| {
                match i.kind {
                    ItemKind::Struct(_, _, ref struct_def)
                    | ItemKind::Union(_, _, ref struct_def) => {
                        // If this is a unit or tuple-like struct, register the constructor.
                        if let Some((ctor_kind, ctor_node_id)) = CtorKind::from_ast(struct_def) {
                            this.create_def(
                                ctor_node_id,
                                None,
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
            FnKind::Fn(
                _ctxt,
                _vis,
                Fn {
                    sig: FnSig { header, decl, span: _ }, ident, generics, contract, body, ..
                },
            ) if let Some(coroutine_kind) = header.coroutine_kind => {
                self.visit_ident(ident);
                self.visit_fn_header(header);
                self.visit_generics(generics);
                if let Some(contract) = contract {
                    self.visit_contract(contract);
                }

                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                let FnDecl { inputs, output } = &**decl;
                for param in inputs {
                    self.visit_param(param);
                }

                let (return_id, return_span) = coroutine_kind.return_id();
                let return_def = self.create_def(return_id, None, DefKind::OpaqueTy, return_span);
                self.with_parent(return_def, |this| this.visit_fn_ret_ty(output));

                // If this async fn has no body (i.e. it's an async fn signature in a trait)
                // then the closure_def will never be used, and we should avoid generating a
                // def-id for it.
                if let Some(body) = body {
                    let closure_def =
                        self.create_def(coroutine_kind.closure_id(), None, DefKind::Closure, span);
                    self.with_parent(closure_def, |this| this.visit_block(body));
                }
            }
            FnKind::Closure(binder, Some(coroutine_kind), decl, body) => {
                self.visit_closure_binder(binder);
                visit::walk_fn_decl(self, decl);

                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let coroutine_def =
                    self.create_def(coroutine_kind.closure_id(), None, DefKind::Closure, span);
                self.with_parent(coroutine_def, |this| this.visit_expr(body));
            }
            _ => visit::walk_fn(self, fn_kind),
        }
    }

    fn visit_nested_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId) {
        self.create_def(id, None, DefKind::Use, use_tree.span);
        visit::walk_use_tree(self, use_tree);
    }

    fn visit_foreign_item(&mut self, fi: &'a ForeignItem) {
        let (ident, def_kind) = match fi.kind {
            ForeignItemKind::Static(box StaticItem {
                ident,
                ty: _,
                mutability,
                expr: _,
                safety,
                define_opaque: _,
            }) => {
                let safety = match safety {
                    ast::Safety::Unsafe(_) | ast::Safety::Default => hir::Safety::Unsafe,
                    ast::Safety::Safe(_) => hir::Safety::Safe,
                };

                (ident, DefKind::Static { safety, mutability, nested: false })
            }
            ForeignItemKind::Fn(box Fn { ident, .. }) => (ident, DefKind::Fn),
            ForeignItemKind::TyAlias(box TyAlias { ident, .. }) => (ident, DefKind::ForeignTy),
            ForeignItemKind::MacCall(_) => return self.visit_macro_invoc(fi.id),
        };

        let def = self.create_def(fi.id, Some(ident.name), def_kind, fi.span);

        self.with_parent(def, |this| visit::walk_item(this, fi));
    }

    fn visit_variant(&mut self, v: &'a Variant) {
        if v.is_placeholder {
            return self.visit_macro_invoc(v.id);
        }
        let def = self.create_def(v.id, Some(v.ident.name), DefKind::Variant, v.span);
        self.with_parent(def, |this| {
            if let Some((ctor_kind, ctor_node_id)) = CtorKind::from_ast(&v.data) {
                this.create_def(
                    ctor_node_id,
                    None,
                    DefKind::Ctor(CtorOf::Variant, ctor_kind),
                    v.span,
                );
            }
            visit::walk_variant(this, v)
        });
    }

    fn visit_where_predicate(&mut self, pred: &'a WherePredicate) {
        if pred.is_placeholder {
            self.visit_macro_invoc(pred.id)
        } else {
            visit::walk_where_predicate(self, pred)
        }
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
        self.create_def(param.id, Some(param.ident.name), def_kind, param.ident.span);

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
        let (ident, def_kind) = match &i.kind {
            AssocItemKind::Fn(box Fn { ident, .. })
            | AssocItemKind::Delegation(box Delegation { ident, .. }) => (*ident, DefKind::AssocFn),
            AssocItemKind::Const(box ConstItem { ident, .. }) => (*ident, DefKind::AssocConst),
            AssocItemKind::Type(box TyAlias { ident, .. }) => (*ident, DefKind::AssocTy),
            AssocItemKind::MacCall(..) | AssocItemKind::DelegationMac(..) => {
                return self.visit_macro_invoc(i.id);
            }
        };

        let def = self.create_def(i.id, Some(ident.name), def_kind, i.span);
        self.with_parent(def, |this| visit::walk_assoc_item(this, i, ctxt));
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.kind {
            PatKind::MacCall(..) => self.visit_macro_invoc(pat.id),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_anon_const(&mut self, constant: &'a AnonConst) {
        let parent = self.create_def(constant.id, None, DefKind::AnonConst, constant.value.span);
        self.with_parent(parent, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = match expr.kind {
            ExprKind::MacCall(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(..) | ExprKind::Gen(..) => {
                self.create_def(expr.id, None, DefKind::Closure, expr.span)
            }
            ExprKind::ConstBlock(ref constant) => {
                for attr in &expr.attrs {
                    visit::walk_attribute(self, attr);
                }
                let def =
                    self.create_def(constant.id, None, DefKind::InlineConst, constant.value.span);
                self.with_parent(def, |this| visit::walk_anon_const(this, constant));
                return;
            }
            _ => self.invocation_parent.parent_def,
        };

        self.with_parent(parent_def, |this| visit::walk_expr(this, expr))
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.kind {
            TyKind::MacCall(..) => self.visit_macro_invoc(ty.id),
            TyKind::ImplTrait(opaque_id, _) => {
                let name = *self
                    .resolver
                    .impl_trait_names
                    .get(&ty.id)
                    .unwrap_or_else(|| span_bug!(ty.span, "expected this opaque to be named"));
                let kind = match self.invocation_parent.impl_trait_context {
                    ImplTraitContext::Universal => DefKind::TyParam,
                    ImplTraitContext::Existential => DefKind::OpaqueTy,
                    ImplTraitContext::InBinding => return visit::walk_ty(self, ty),
                };
                let id = self.create_def(opaque_id, Some(name), kind, ty.span);
                match self.invocation_parent.impl_trait_context {
                    // Do not nest APIT, as we desugar them as `impl_trait: bounds`,
                    // so the `impl_trait` node is not a parent to `bounds`.
                    ImplTraitContext::Universal => visit::walk_ty(self, ty),
                    ImplTraitContext::Existential => {
                        self.with_parent(id, |this| visit::walk_ty(this, ty))
                    }
                    ImplTraitContext::InBinding => unreachable!(),
                };
            }
            _ => visit::walk_ty(self, ty),
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt.kind {
            StmtKind::MacCall(..) => self.visit_macro_invoc(stmt.id),
            // FIXME(impl_trait_in_bindings): We don't really have a good way of
            // introducing the right `ImplTraitContext` here for all the cases we
            // care about, in case we want to introduce ITIB to other positions
            // such as turbofishes (e.g. `foo::<impl Fn()>(|| {})`).
            StmtKind::Let(ref local) => self.with_impl_trait(ImplTraitContext::InBinding, |this| {
                visit::walk_local(this, local)
            }),
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
        let orig_in_attr = mem::replace(&mut self.invocation_parent.in_attr, true);
        visit::walk_attribute(self, attr);
        self.invocation_parent.in_attr = orig_in_attr;
    }

    fn visit_inline_asm(&mut self, asm: &'a InlineAsm) {
        let InlineAsm {
            asm_macro: _,
            template: _,
            template_strs: _,
            operands,
            clobber_abis: _,
            options: _,
            line_spans: _,
        } = asm;
        for (op, _span) in operands {
            match op {
                InlineAsmOperand::In { expr, reg: _ }
                | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
                | InlineAsmOperand::InOut { expr, reg: _, late: _ } => {
                    self.visit_expr(expr);
                }
                InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
                InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                    self.visit_expr(in_expr);
                    if let Some(expr) = out_expr {
                        self.visit_expr(expr);
                    }
                }
                InlineAsmOperand::Const { anon_const } => {
                    let def = self.create_def(
                        anon_const.id,
                        None,
                        DefKind::InlineConst,
                        anon_const.value.span,
                    );
                    self.with_parent(def, |this| visit::walk_anon_const(this, anon_const));
                }
                InlineAsmOperand::Sym { sym } => self.visit_inline_asm_sym(sym),
                InlineAsmOperand::Label { block } => self.visit_block(block),
            }
        }
    }
}
