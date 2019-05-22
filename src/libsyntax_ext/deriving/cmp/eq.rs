use crate::deriving::path_std;
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use syntax::ast::{self, Expr, MetaItem, GenericArg};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::{sym, Symbol};
use syntax_pos::Span;

pub fn expand_deriving_eq(cx: &mut ExtCtxt<'_>,
                          span: Span,
                          mitem: &MetaItem,
                          item: &Annotatable,
                          push: &mut dyn FnMut(Annotatable)) {
    let inline = cx.meta_word(span, sym::inline);
    let hidden = cx.meta_list_item_word(span, sym::hidden);
    let doc = cx.meta_list(span, sym::doc, vec![hidden]);
    let attrs = vec![cx.attribute(span, inline), cx.attribute(span, doc)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cx, cmp::Eq),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: vec![MethodDef {
                          name: "assert_receiver_is_total_eq",
                          generics: LifetimeBounds::empty(),
                          explicit_self: borrowed_explicit_self(),
                          args: vec![],
                          ret_ty: nil_ty(),
                          attributes: attrs,
                          is_unsafe: false,
                          unify_fieldless_variants: true,
                          combine_substructure: combine_substructure(Box::new(|a, b, c| {
                              cs_total_eq_assert(a, b, c)
                          })),
                      }],
        associated_types: Vec::new(),
    };
    trait_def.expand_ext(cx, mitem, item, push, true)
}

fn cs_total_eq_assert(cx: &mut ExtCtxt<'_>,
                      trait_span: Span,
                      substr: &Substructure<'_>)
                      -> P<Expr> {
    fn assert_ty_bounds(cx: &mut ExtCtxt<'_>, stmts: &mut Vec<ast::Stmt>,
                        ty: P<ast::Ty>, span: Span, helper_name: &str) {
        // Generate statement `let _: helper_name<ty>;`,
        // set the expn ID so we can use the unstable struct.
        let span = span.with_ctxt(cx.backtrace());
        let assert_path = cx.path_all(span, true,
                                        cx.std_path(&[sym::cmp, Symbol::intern(helper_name)]),
                                        vec![GenericArg::Type(ty)], vec![]);
        stmts.push(cx.stmt_let_type_only(span, cx.ty_path(assert_path)));
    }
    fn process_variant(cx: &mut ExtCtxt<'_>,
                       stmts: &mut Vec<ast::Stmt>,
                       variant: &ast::VariantData) {
        for field in variant.fields() {
            // let _: AssertParamIsEq<FieldTy>;
            assert_ty_bounds(cx, stmts, field.ty.clone(), field.span, "AssertParamIsEq");
        }
    }

    let mut stmts = Vec::new();
    match *substr.fields {
        StaticStruct(vdata, ..) => {
            process_variant(cx, &mut stmts, vdata);
        }
        StaticEnum(enum_def, ..) => {
            for variant in &enum_def.variants {
                process_variant(cx, &mut stmts, &variant.node.data);
            }
        }
        _ => cx.span_bug(trait_span, "unexpected substructure in `derive(Eq)`")
    }
    cx.expr_block(cx.block(trait_span, stmts))
}
