use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::{self as ast, MetaItem, Safety};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::path_std;

struct ReplaceSelfTyVisitor(Box<ast::Ty>);
impl MutVisitor for ReplaceSelfTyVisitor {
    fn visit_ty(&mut self, ty: &mut ast::Ty) {
        if let ast::TyKind::Path(None, path) = &mut ty.kind
            && let [first, rest @ ..] = &path.segments[..]
            && *first == kw::SelfUpper
        {
            if rest.is_empty() {
                // Just `Self` — replace the whole type
                *ty = *self.0.clone();
            } else {
                // `Self::Something` — splice concrete type's segments in
                let ast::TyKind::Path(_, concrete_path) = &self.0.kind else {
                    unreachable!("expected Self type to be a path");
                };
                let mut new_segments = concrete_path.segments.clone();
                new_segments.extend_from_slice(rest);
                path.segments = new_segments;
                mut_visit::walk_ty(self, ty);
            }
        } else {
            mut_visit::walk_ty(self, ty);
        }
    }
    fn visit_expr(&mut self, expr: &mut ast::Expr) {
        if let ast::ExprKind::Path(None, path) = &mut expr.kind
            && let [first, rest @ ..] = &*path.segments
            && *first == kw::SelfUpper
        {
            let ast::TyKind::Path(_, concrete_path) = &self.0.kind else {
                unreachable!("expected Self type to be a path");
            };
            let mut new_segments = concrete_path.segments.clone();
            new_segments.extend_from_slice(rest);
            path.segments = new_segments;
        }
        mut_visit::walk_expr(self, expr);
    }
}

struct RespanGenericsVisitor(Span);
impl MutVisitor for RespanGenericsVisitor {
    fn visit_generics(&mut self, generics: &mut ast::Generics) {
        generics.where_clause.span = self.0.with_ctxt(generics.where_clause.span.ctxt());
        generics.span = self.0.with_ctxt(generics.span.ctxt());
        // generic parameter declarations don't need to be respanned, so we visit the where clause
        // predicates next
        for predicate in &mut generics.where_clause.predicates {
            self.visit_where_predicate(predicate);
        }
    }
    fn visit_where_predicate(&mut self, predicate: &mut ast::WherePredicate) {
        predicate.span = self.0.with_ctxt(predicate.span.ctxt());
        mut_visit::walk_where_predicate(self, predicate);
    }
    fn visit_where_predicate_kind(&mut self, kind: &mut ast::WherePredicateKind) {
        match kind {
            ast::WherePredicateKind::BoundPredicate(bound_predicate) => {
                bound_predicate.bounded_ty.span =
                    self.0.with_ctxt(bound_predicate.bounded_ty.span.ctxt());
            }
            ast::WherePredicateKind::EqPredicate(eq_predicate) => {
                eq_predicate.lhs_ty.span = self.0.with_ctxt(eq_predicate.lhs_ty.span.ctxt());
                eq_predicate.rhs_ty.span = self.0.with_ctxt(eq_predicate.rhs_ty.span.ctxt());
            }
            ast::WherePredicateKind::RegionPredicate(_) => {}
        }
        mut_visit::walk_where_predicate_kind(self, kind);
    }
    fn visit_param_bound(
        &mut self,
        bound: &mut rustc_ast::GenericBound,
        _ctxt: rustc_ast::visit::BoundKind,
    ) {
        match bound {
            ast::GenericBound::Trait(poly_trait_ref) => {
                poly_trait_ref.span = self.0.with_ctxt(poly_trait_ref.span.ctxt());
            }
            ast::GenericBound::Outlives(_) => {}
            ast::GenericBound::Use(_, _) => {}
        }
        ast::mut_visit::walk_param_bound(self, bound);
    }
}

struct StripConstTraitBoundsVisitor;
impl MutVisitor for StripConstTraitBoundsVisitor {
    fn visit_param_bound(
        &mut self,
        bound: &mut rustc_ast::GenericBound,
        _ctxt: rustc_ast::visit::BoundKind,
    ) {
        if let ast::GenericBound::Trait(poly_trait_ref) = bound {
            poly_trait_ref.modifiers.constness = ast::BoundConstness::Never;
        }
        mut_visit::walk_param_bound(self, bound);
    }
}

pub(crate) fn expand_deriving_eq(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let span = cx.with_def_site_ctxt(span);

    let mut fn_generics = ast::Generics { span, ..Default::default() };
    let mut self_ty = None;

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::Eq),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const,
        is_staged_api_crate: cx.ecfg.features.staged_api(),
        safety: Safety::Default,
        document: true,
    };
    trait_def.expand_ext(
        cx,
        mitem,
        item,
        &mut |mut a| {
            let Annotatable::Item(item) = &mut a else {
                unreachable!("should have emitted an Item in trait_def.expand_ext");
            };
            let ast::ItemKind::Impl(imp) = &mut item.kind else {
                unreachable!("should have emitted an Impl in trait_def.expand_ext");
            };
            use ast::mut_visit::MutVisitor;
            RespanGenericsVisitor(span).visit_generics(&mut imp.generics);
            fn_generics = imp.generics.clone();
            self_ty = Some(imp.self_ty.clone());
            push(a)
        },
        true,
    );

    let self_ty =
        self_ty.unwrap_or_else(|| cx.dcx().span_bug(span, "missing self type in `derive(Eq)`"));
    let assert_stmts =
        eq_assert_stmts_from_item(cx, span, item, ReplaceSelfTyVisitor(self_ty.clone()));

    // Skip generating `assert_fields_are_eq` impl if there are no assertions to make
    if assert_stmts.is_empty() {
        return;
    }

    StripConstTraitBoundsVisitor.visit_generics(&mut fn_generics);
    push(Annotatable::Item(expand_const_item_block(cx, span, fn_generics, self_ty, assert_stmts)));
}

fn expand_const_item_block(
    cx: &ExtCtxt<'_>,
    span: Span,
    fn_generics: ast::Generics,
    self_ty: Box<ast::Ty>,
    assert_stmts: ThinVec<ast::Stmt>,
) -> Box<ast::Item> {
    // We need a dummy const pointer to Self argument to ensure well-formedness of the Self type.
    // This doesn't add overhead because the fn itself is never called, and in fact should not
    // even have any runtime code generated for it as it's an inline const fn.
    let const_self_ptr_ty =
        cx.ty(span, ast::TyKind::Ptr(ast::MutTy { mutbl: ast::Mutability::Not, ty: self_ty }));
    let fn_args = thin_vec![cx.param(span, Ident::new(kw::Underscore, span), const_self_ptr_ty)];
    let fn_sig = ast::FnSig {
        header: ast::FnHeader {
            constness: ast::Const::Yes(span),
            coroutine_kind: None,
            safety: ast::Safety::Default,
            ext: ast::Extern::None,
        },
        decl: cx.fn_decl(fn_args, ast::FnRetTy::Default(span)),
        span,
    };

    cx.item(
        span,
        ast::AttrVec::new(),
        ast::ItemKind::ConstBlock(ast::ConstBlockItem {
            span,
            id: ast::DUMMY_NODE_ID,
            block: cx.block(
                span,
                thin_vec![cx.stmt_item(
                    span,
                    Box::new(ast::Item {
                        span,
                        id: ast::DUMMY_NODE_ID,
                        attrs: thin_vec![
                            cx.attr_nested_word(sym::doc, sym::hidden, span),
                            cx.attr_nested_word(sym::coverage, sym::off, span),
                            // This function will never be called, so doing codegen etc. for it is
                            // unnecessary. We prevent this by adding `#[inline]`, which improves
                            // compile-time.
                            cx.attr_word(sym::inline, span),
                        ],
                        vis: ast::Visibility {
                            kind: ast::VisibilityKind::Inherited,
                            span,
                            tokens: None,
                        },
                        tokens: None,
                        kind: ast::ItemKind::Fn(Box::new(ast::Fn {
                            defaultness: ast::Defaultness::Implicit,
                            ident: Ident::new(sym::assert_fields_are_eq, span),
                            generics: fn_generics,
                            sig: fn_sig,
                            contract: None,
                            define_opaque: None,
                            body: Some(cx.block(span, assert_stmts)),
                            eii_impls: ThinVec::new(),
                        }))
                    })
                ),],
            ),
        }),
    )
}

fn eq_assert_stmts_from_item(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &Annotatable,
    mut replace_self_ty: ReplaceSelfTyVisitor,
) -> ThinVec<ast::Stmt> {
    let mut stmts = ThinVec::new();
    let mut seen_type_names = FxHashSet::default();
    let mut process_variant = |variant: &ast::VariantData| {
        for field in variant.fields() {
            // This basic redundancy checking only prevents duplication of
            // assertions like `AssertParamIsEq<Foo>` where the type is a
            // simple name. That's enough to get a lot of cases, though.
            if let Some(name) = field.ty.kind.is_simple_path()
                && !seen_type_names.insert(name)
            {
                // Already produced an assertion for this type.
            } else {
                use ast::mut_visit::MutVisitor;
                let mut field_ty = field.ty.clone();
                replace_self_ty.visit_ty(&mut field_ty);
                // let _: AssertParamIsEq<FieldTy>;
                super::assert_ty_bounds(
                    cx,
                    &mut stmts,
                    field_ty,
                    field.span,
                    &[sym::cmp, sym::AssertParamIsEq],
                );
            }
        }
    };
    match item {
        Annotatable::Item(item) => match &item.kind {
            ast::ItemKind::Struct(_, _, vdata) => {
                process_variant(vdata);
            }
            ast::ItemKind::Enum(_, _, enum_def) => {
                for variant in &enum_def.variants {
                    process_variant(&variant.data);
                }
            }
            ast::ItemKind::Union(_, _, vdata) => {
                process_variant(vdata);
            }
            _ => cx.dcx().span_bug(span, "unexpected item in `derive(Eq)`"),
        },
        _ => cx.dcx().span_bug(span, "unexpected item in `derive(Eq)`"),
    }
    stmts
}
