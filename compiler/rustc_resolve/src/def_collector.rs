use std::mem;

use rustc_ast::mut_visit::FnKind;
use rustc_ast::visit::AssocCtxt;
use rustc_ast::*;
use rustc_attr_parsing::{AttributeParser, Early, OmitDoc, ShouldEmit};
use rustc_expand::expand::AstFragment;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::span_bug;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::{Span, Symbol, sym};
use smallvec::{SmallVec, smallvec};
use tracing::debug;

use crate::{ImplTraitContext, InvocationParent, Resolver};

#[tracing::instrument(level = "trace", skip(resolver, fragment), ret)]
pub(crate) fn collect_definitions(
    resolver: &mut Resolver<'_, '_>,
    fragment: &mut AstFragment,
    expansion: LocalExpnId,
) {
    let invocation_parent = resolver.invocation_parents[&expansion];
    let mut visitor = DefCollector { resolver, expansion, invocation_parent };
    fragment.mut_visit_with(&mut visitor);
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

    fn with_parent(&mut self, parent_def: LocalDefId, f: impl FnOnce(&mut Self)) {
        let orig_parent_def = mem::replace(&mut self.invocation_parent.parent_def, parent_def);
        f(self);
        self.invocation_parent.parent_def = orig_parent_def;
    }

    fn with_impl_trait(&mut self, impl_trait_context: ImplTraitContext, f: impl FnOnce(&mut Self)) {
        let orig_itc =
            mem::replace(&mut self.invocation_parent.impl_trait_context, impl_trait_context);
        f(self);
        self.invocation_parent.impl_trait_context = orig_itc;
    }

    fn collect_field(&mut self, field: &mut FieldDef, index: Option<usize>) {
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
            self.with_parent(def, |this| mut_visit::walk_field_def(this, field))
        }
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_macro_invoc(&mut self, id: NodeId) {
        let id = id.placeholder_to_expn_id();
        let old_parent = self.resolver.invocation_parents.insert(id, self.invocation_parent);
        assert!(old_parent.is_none(), "parent `LocalDefId` is reset for an invocation");
    }
}

impl<'a, 'ra, 'tcx> mut_visit::MutVisitor for DefCollector<'a, 'ra, 'tcx> {
    fn visit_span(&mut self, span: &mut Span) {
        if self.resolver.tcx.sess.opts.incremental.is_some() {
            *span = span.with_parent(Some(self.invocation_parent.parent_def));
        }
    }

    fn visit_item(&mut self, i: &mut Item) {
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
                let macro_kind = macro_data.ext.macro_kind();
                opt_macro_data = Some(macro_data);
                DefKind::Macro(macro_kind)
            }
            ItemKind::GlobalAsm(..) => DefKind::GlobalAsm,
            ItemKind::Use(use_tree) => {
                self.create_def(i.id, None, DefKind::Use, use_tree.span);
                // HIR lowers use trees as a flat stream of `ItemKind::Use`.
                // This means all the def-ids must be parented to the module.
                return mut_visit::walk_item(self, i);
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
                mut_visit::walk_item(this, i)
            })
        })
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_fn(&mut self, mut fn_kind: FnKind<'_>, fn_span: Span, _: NodeId) {
        match &mut fn_kind {
            FnKind::Fn(
                _ctxt,
                _vis,
                Fn {
                    defaultness,
                    ident,
                    sig: FnSig { header, decl, span },
                    generics,
                    contract,
                    body,
                    define_opaque,
                },
            ) => {
                mut_visit::walk_defaultness(self, defaultness);
                self.visit_ident(ident);
                self.visit_fn_header(header);
                self.visit_generics(generics);
                if let Some(contract) = contract {
                    self.visit_contract(contract);
                }
                self.visit_span(span);
                if let Some(define_opaque) = define_opaque {
                    for (node_id, path) in define_opaque {
                        self.visit_id(node_id);
                        self.visit_path(path);
                    }
                }

                // For async functions, we need to create their inner defs inside of a
                // closure to match their desugared representation. Besides that,
                // we must mirror everything that `visit::walk_fn` below does.
                let FnDecl { inputs, output } = &mut **decl;
                for param in inputs {
                    self.visit_param(param);
                }

                let return_def = if let Some(coroutine_kind) = header.coroutine_kind {
                    // coroutine_kind has been visited by visit_header.
                    let (return_id, return_span) = coroutine_kind.return_id();
                    self.create_def(return_id, None, DefKind::OpaqueTy, return_span)
                } else {
                    self.invocation_parent.parent_def
                };
                self.with_parent(return_def, |this| this.visit_fn_ret_ty(output));

                // If this async fn has no body (i.e. it's an async fn signature in a trait)
                // then the closure_def will never be used, and we should avoid generating a
                // def-id for it.
                if let Some(body) = body {
                    let closure_def = if let Some(coroutine_kind) = header.coroutine_kind {
                        self.create_def(
                            coroutine_kind.closure_id(),
                            None,
                            DefKind::Closure,
                            fn_span,
                        )
                    } else {
                        self.invocation_parent.parent_def
                    };
                    self.with_parent(closure_def, |this| this.visit_block(body));
                }
            }
            FnKind::Closure(binder, coroutine_kind, decl, body) => {
                self.visit_closure_binder(binder);
                self.visit_fn_decl(decl);

                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                let closure_def = if let Some(coroutine_kind) = coroutine_kind {
                    self.visit_coroutine_kind(coroutine_kind);
                    self.create_def(coroutine_kind.closure_id(), None, DefKind::Closure, fn_span)
                } else {
                    self.invocation_parent.parent_def
                };
                self.with_parent(closure_def, |this| this.visit_expr(body));
            }
        }
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
                    // This means all the def-ids must be parented to the module.
                    self.create_def(*id, None, DefKind::Use, *span);
                    self.visit_use_tree(tree);
                }
                self.visit_span(span);
            }
            UseTreeKind::Glob => {}
        }
        self.visit_span(span);
    }

    fn visit_foreign_item(&mut self, fi: &mut ForeignItem) {
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
        self.with_parent(def, |this| mut_visit::walk_item(this, fi))
    }

    fn visit_variant(&mut self, v: &mut Variant) {
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
            mut_visit::walk_variant(this, v)
        })
    }

    fn visit_where_predicate(&mut self, pred: &mut WherePredicate) {
        if pred.is_placeholder {
            self.visit_macro_invoc(pred.id)
        } else {
            mut_visit::walk_where_predicate(self, pred)
        }
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
        for (index, field) in fields.iter_mut().enumerate() {
            self.collect_field(field, Some(index));
        }
    }

    fn visit_generic_param(&mut self, param: &mut GenericParam) {
        if param.is_placeholder {
            return self.visit_macro_invoc(param.id);
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
            mut_visit::walk_generic_param(this, param)
        })
    }

    fn visit_assoc_item(&mut self, i: &mut AssocItem, ctxt: AssocCtxt) {
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
        self.with_parent(def, |this| mut_visit::walk_assoc_item(this, i, ctxt));
    }

    fn visit_pat(&mut self, pat: &mut Pat) {
        if let PatKind::MacCall(..) = pat.kind {
            return self.visit_macro_invoc(pat.id);
        }
        mut_visit::walk_pat(self, pat)
    }

    fn visit_anon_const(&mut self, constant: &mut AnonConst) {
        let parent = self.create_def(constant.id, None, DefKind::AnonConst, constant.value.span);
        self.with_parent(parent, |this| mut_visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &mut Expr) {
        let parent_def = match expr.kind {
            ExprKind::MacCall(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(..) | ExprKind::Gen(..) => {
                self.create_def(expr.id, None, DefKind::Closure, expr.span)
            }
            ExprKind::ConstBlock(ref mut constant) => {
                let Expr { id: _, kind: _, span, attrs, tokens: _ } = expr;
                for attr in attrs {
                    self.visit_attribute(attr);
                }
                let def =
                    self.create_def(constant.id, None, DefKind::InlineConst, constant.value.span);
                self.with_parent(def, |this| mut_visit::walk_anon_const(this, constant));
                self.visit_span(span);
                return;
            }
            _ => self.invocation_parent.parent_def,
        };

        self.with_parent(parent_def, |this| mut_visit::walk_expr(this, expr))
    }

    fn visit_ty(&mut self, ty: &mut Ty) {
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
                    ImplTraitContext::InBinding => return mut_visit::walk_ty(self, ty),
                };
                let id = self.create_def(opaque_id, Some(name), kind, ty.span);
                match self.invocation_parent.impl_trait_context {
                    // Do not nest APIT, as we desugar them as `impl_trait: bounds`,
                    // so the `impl_trait` node is not a parent to `bounds`.
                    ImplTraitContext::Universal => mut_visit::walk_ty(self, ty),
                    ImplTraitContext::Existential => {
                        self.with_parent(id, |this| mut_visit::walk_ty(this, ty))
                    }
                    ImplTraitContext::InBinding => unreachable!(),
                };
            }
            _ => mut_visit::walk_ty(self, ty),
        }
    }

    fn flat_map_stmt(&mut self, mut stmt: Stmt) -> SmallVec<[Stmt; 1]> {
        let Stmt { id, kind, span } = &mut stmt;
        match kind {
            StmtKind::MacCall(..) => {
                self.visit_macro_invoc(*id);
                smallvec![stmt]
            }
            // FIXME(impl_trait_in_bindings): We don't really have a good way of
            // introducing the right `ImplTraitContext` here for all the cases we
            // care about, in case we want to introduce ITIB to other positions
            // such as turbofishes (e.g. `foo::<impl Fn()>(|| {})`).
            StmtKind::Let(local) => {
                self.with_impl_trait(ImplTraitContext::InBinding, |this| {
                    mut_visit::walk_local(this, local);
                    this.visit_span(span);
                });
                smallvec![stmt]
            }
            _ => mut_visit::walk_flat_map_stmt(self, stmt),
        }
    }

    fn visit_arm(&mut self, arm: &mut Arm) {
        if arm.is_placeholder {
            return self.visit_macro_invoc(arm.id);
        }
        mut_visit::walk_arm(self, arm)
    }

    fn visit_expr_field(&mut self, f: &mut ExprField) {
        if f.is_placeholder {
            return self.visit_macro_invoc(f.id);
        }
        mut_visit::walk_expr_field(self, f)
    }

    fn visit_pat_field(&mut self, fp: &mut PatField) {
        if fp.is_placeholder {
            return self.visit_macro_invoc(fp.id);
        }
        mut_visit::walk_pat_field(self, fp)
    }

    fn visit_param(&mut self, p: &mut Param) {
        if p.is_placeholder {
            return self.visit_macro_invoc(p.id);
        }
        self.with_impl_trait(ImplTraitContext::Universal, |this| mut_visit::walk_param(this, p))
    }

    // This method is called only when we are visiting an individual field
    // after expanding an attribute on it.
    fn visit_field_def(&mut self, field: &mut FieldDef) {
        self.collect_field(field, None)
    }

    fn visit_crate(&mut self, krate: &mut Crate) {
        if krate.is_placeholder {
            return self.visit_macro_invoc(krate.id);
        }
        mut_visit::walk_crate(self, krate)
    }

    fn visit_attribute(&mut self, attr: &mut Attribute) {
        let orig_in_attr = mem::replace(&mut self.invocation_parent.in_attr, true);
        mut_visit::walk_attribute(self, attr);
        self.invocation_parent.in_attr = orig_in_attr;
    }

    fn visit_inline_asm(&mut self, asm: &mut InlineAsm) {
        let InlineAsm {
            asm_macro: _,
            template,
            template_strs,
            operands,
            clobber_abis,
            options: _,
            line_spans,
        } = asm;
        for piece in template {
            match piece {
                InlineAsmTemplatePiece::String(_str) => {}
                InlineAsmTemplatePiece::Placeholder { operand_idx: _, modifier: _, span } => {
                    self.visit_span(span);
                }
            }
        }
        for (_s1, _s2, span) in template_strs {
            self.visit_span(span);
        }
        for (op, span) in operands {
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
                    self.with_parent(def, |this| mut_visit::walk_anon_const(this, anon_const));
                }
                InlineAsmOperand::Sym { sym } => self.visit_inline_asm_sym(sym),
                InlineAsmOperand::Label { block } => self.visit_block(block),
            }
            self.visit_span(span);
        }
        for (_s1, span) in clobber_abis {
            self.visit_span(span);
        }
        for span in line_spans {
            self.visit_span(span);
        }
    }
}
