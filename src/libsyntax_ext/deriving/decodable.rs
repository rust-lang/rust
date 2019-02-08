//! The compiler code necessary for `#[derive(Decodable)]`. See encodable.rs for more.

use crate::deriving::{self, pathvec_std};
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;
use crate::deriving::warn_if_deprecated;

use syntax::ast;
use syntax::ast::{Expr, MetaItem, Mutability};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;

pub fn expand_deriving_rustc_decodable(cx: &mut ExtCtxt<'_>,
                                       span: Span,
                                       mitem: &MetaItem,
                                       item: &Annotatable,
                                       push: &mut dyn FnMut(Annotatable)) {
    expand_deriving_decodable_imp(cx, span, mitem, item, push, "rustc_serialize")
}

pub fn expand_deriving_decodable(cx: &mut ExtCtxt<'_>,
                                 span: Span,
                                 mitem: &MetaItem,
                                 item: &Annotatable,
                                 push: &mut dyn FnMut(Annotatable)) {
    warn_if_deprecated(cx, span, "Decodable");
    expand_deriving_decodable_imp(cx, span, mitem, item, push, "serialize")
}

fn expand_deriving_decodable_imp(cx: &mut ExtCtxt<'_>,
                                 span: Span,
                                 mitem: &MetaItem,
                                 item: &Annotatable,
                                 push: &mut dyn FnMut(Annotatable),
                                 krate: &'static str) {
    let typaram = &*deriving::hygienic_type_parameter(item, "__D");

    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: Path::new_(vec![krate, "Decodable"], None, vec![], PathKind::Global),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
                          name: "decode",
                          generics: LifetimeBounds {
                              lifetimes: Vec::new(),
                              bounds: vec![(typaram,
                                            vec![Path::new_(vec![krate, "Decoder"],
                                                            None,
                                                            vec![],
                                                            PathKind::Global)])],
                          },
                          explicit_self: None,
                          args: vec![(Ptr(Box::new(Literal(Path::new_local(typaram))),
                                         Borrowed(None, Mutability::Mutable)), "d")],
                          ret_ty:
                              Literal(Path::new_(pathvec_std!(cx, result::Result),
                                                 None,
                                                 vec![Box::new(Self_), Box::new(Literal(Path::new_(
                        vec![typaram, "Error"], None, vec![], PathKind::Local
                    )))],
                                                 PathKind::Std)),
                          attributes: Vec::new(),
                          is_unsafe: false,
                          unify_fieldless_variants: false,
                          combine_substructure: combine_substructure(Box::new(|a, b, c| {
                              decodable_substructure(a, b, c, krate)
                          })),
                      }],
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}

fn decodable_substructure(cx: &mut ExtCtxt<'_>,
                          trait_span: Span,
                          substr: &Substructure<'_>,
                          krate: &str)
                          -> P<Expr> {
    let decoder = substr.nonself_args[0].clone();
    let recurse = vec![cx.ident_of(krate), cx.ident_of("Decodable"), cx.ident_of("decode")];
    let exprdecode = cx.expr_path(cx.path_global(trait_span, recurse));
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_d");
    let blkdecoder = cx.expr_ident(trait_span, blkarg);

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            let nfields = match *summary {
                Unnamed(ref fields, _) => fields.len(),
                Named(ref fields) => fields.len(),
            };
            let read_struct_field = cx.ident_of("read_struct_field");

            let path = cx.path_ident(trait_span, substr.type_ident);
            let result =
                decode_static_fields(cx, trait_span, path, summary, |cx, span, name, field| {
                    cx.expr_try(span,
                                cx.expr_method_call(span,
                                                    blkdecoder.clone(),
                                                    read_struct_field,
                                                    vec![cx.expr_str(span, name),
                                                         cx.expr_usize(span, field),
                                                         exprdecode.clone()]))
                });
            let result = cx.expr_ok(trait_span, result);
            cx.expr_method_call(trait_span,
                                decoder,
                                cx.ident_of("read_struct"),
                                vec![cx.expr_str(trait_span, substr.type_ident.name),
                                     cx.expr_usize(trait_span, nfields),
                                     cx.lambda1(trait_span, result, blkarg)])
        }
        StaticEnum(_, ref fields) => {
            let variant = cx.ident_of("i");

            let mut arms = Vec::with_capacity(fields.len() + 1);
            let mut variants = Vec::with_capacity(fields.len());
            let rvariant_arg = cx.ident_of("read_enum_variant_arg");

            for (i, &(ident, v_span, ref parts)) in fields.iter().enumerate() {
                variants.push(cx.expr_str(v_span, ident.name));

                let path = cx.path(trait_span, vec![substr.type_ident, ident]);
                let decoded = decode_static_fields(cx, v_span, path, parts, |cx, span, _, field| {
                    let idx = cx.expr_usize(span, field);
                    cx.expr_try(span,
                                cx.expr_method_call(span,
                                                    blkdecoder.clone(),
                                                    rvariant_arg,
                                                    vec![idx, exprdecode.clone()]))
                });

                arms.push(cx.arm(v_span,
                                 vec![cx.pat_lit(v_span, cx.expr_usize(v_span, i))],
                                 decoded));
            }

            arms.push(cx.arm_unreachable(trait_span));

            let result =
                cx.expr_ok(trait_span,
                           cx.expr_match(trait_span, cx.expr_ident(trait_span, variant), arms));
            let lambda = cx.lambda(trait_span, vec![blkarg, variant], result);
            let variant_vec = cx.expr_vec(trait_span, variants);
            let variant_vec = cx.expr_addr_of(trait_span, variant_vec);
            let result = cx.expr_method_call(trait_span,
                                             blkdecoder,
                                             cx.ident_of("read_enum_variant"),
                                             vec![variant_vec, lambda]);
            cx.expr_method_call(trait_span,
                                decoder,
                                cx.ident_of("read_enum"),
                                vec![cx.expr_str(trait_span, substr.type_ident.name),
                                     cx.lambda1(trait_span, result, blkarg)])
        }
        _ => cx.bug("expected StaticEnum or StaticStruct in derive(Decodable)"),
    };
}

/// Creates a decoder for a single enum variant/struct:
/// - `outer_pat_path` is the path to this enum variant/struct
/// - `getarg` should retrieve the `usize`-th field with name `@str`.
fn decode_static_fields<F>(cx: &mut ExtCtxt<'_>,
                           trait_span: Span,
                           outer_pat_path: ast::Path,
                           fields: &StaticFields,
                           mut getarg: F)
                           -> P<Expr>
    where F: FnMut(&mut ExtCtxt<'_>, Span, Symbol, usize) -> P<Expr>
{
    match *fields {
        Unnamed(ref fields, is_tuple) => {
            let path_expr = cx.expr_path(outer_pat_path);
            if !is_tuple {
                path_expr
            } else {
                let fields = fields.iter()
                    .enumerate()
                    .map(|(i, &span)| {
                        getarg(cx, span, Symbol::intern(&format!("_field{}", i)), i)
                    })
                    .collect();

                cx.expr_call(trait_span, path_expr, fields)
            }
        }
        Named(ref fields) => {
            // use the field's span to get nicer error messages.
            let fields = fields.iter()
                .enumerate()
                .map(|(i, &(ident, span))| {
                    let arg = getarg(cx, span, ident.name, i);
                    cx.field_imm(span, ident, arg)
                })
                .collect();
            cx.expr_struct(trait_span, outer_pat_path, fields)
        }
    }
}
