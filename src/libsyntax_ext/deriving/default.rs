use crate::deriving::path_std;
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use syntax::ast::{Expr, MetaItem};
use syntax::ext::base::{Annotatable, DummyResult, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::{kw, sym};
use syntax::span_err;
use syntax_pos::Span;

pub fn expand_deriving_default(cx: &mut ExtCtxt<'_>,
                               span: Span,
                               mitem: &MetaItem,
                               item: &Annotatable,
                               push: &mut dyn FnMut(Annotatable)) {
    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(span, inline)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cx, default::Default),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
                          name: "default",
                          generics: LifetimeBounds::empty(),
                          explicit_self: None,
                          args: Vec::new(),
                          ret_ty: Self_,
                          attributes: attrs,
                          is_unsafe: false,
                          unify_fieldless_variants: false,
                          combine_substructure: combine_substructure(Box::new(|a, b, c| {
                              default_substructure(a, b, c)
                          })),
                      }],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

fn default_substructure(cx: &mut ExtCtxt<'_>,
                        trait_span: Span,
                        substr: &Substructure<'_>)
                        -> P<Expr> {
    // Note that `kw::Default` is "default" and `sym::Default` is "Default"!
    let default_ident = cx.std_path(&[kw::Default, sym::Default, kw::Default]);
    let default_call = |span| cx.expr_call_global(span, default_ident.clone(), Vec::new());

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            match *summary {
                Unnamed(ref fields, is_tuple) => {
                    if !is_tuple {
                        cx.expr_ident(trait_span, substr.type_ident)
                    } else {
                        let exprs = fields.iter().map(|sp| default_call(*sp)).collect();
                        cx.expr_call_ident(trait_span, substr.type_ident, exprs)
                    }
                }
                Named(ref fields) => {
                    let default_fields = fields.iter()
                        .map(|&(ident, span)| cx.field_imm(span, ident, default_call(span)))
                        .collect();
                    cx.expr_struct_ident(trait_span, substr.type_ident, default_fields)
                }
            }
        }
        StaticEnum(..) => {
            span_err!(cx, trait_span, E0665,
                      "`Default` cannot be derived for enums, only structs");
            // let compilation continue
            DummyResult::raw_expr(trait_span, true)
        }
        _ => cx.span_bug(trait_span, "Non-static method in `derive(Default)`"),
    };
}
