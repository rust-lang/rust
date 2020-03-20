//! The compiler code necessary for `#[derive(RustcDecodable)]`. See encodable.rs for more.

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::pathvec_std;

use rustc_ast::ast;
use rustc_ast::ast::{Expr, MetaItem, Mutability};
use rustc_ast::ptr::P;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

pub fn expand_deriving_rustc_decodable(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let krate = "rustc_serialize";
    let typaram = "__D";

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
                bounds: vec![(
                    typaram,
                    vec![Path::new_(vec![krate, "Decoder"], None, vec![], PathKind::Global)],
                )],
            },
            explicit_self: None,
            args: vec![(
                Ptr(Box::new(Literal(Path::new_local(typaram))), Borrowed(None, Mutability::Mut)),
                "d",
            )],
            ret_ty: Literal(Path::new_(
                pathvec_std!(cx, result::Result),
                None,
                vec![
                    Box::new(Self_),
                    Box::new(Literal(Path::new_(
                        vec![typaram, "Error"],
                        None,
                        vec![],
                        PathKind::Local,
                    ))),
                ],
                PathKind::Std,
            )),
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

fn decodable_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    krate: &str,
) -> P<Expr> {
    let decoder = substr.nonself_args[0].clone();
    let recurse = vec![
        cx.ident_of(krate, trait_span),
        cx.ident_of("Decodable", trait_span),
        cx.ident_of("decode", trait_span),
    ];
    let exprdecode = cx.expr_path(cx.path_global(trait_span, recurse));
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_d", trait_span);
    let blkdecoder = cx.expr_ident(trait_span, blkarg);

    match *substr.fields {
        StaticStruct(_, ref summary) => {
            let nfields = match *summary {
                Unnamed(ref fields, _) => fields.len(),
                Named(ref fields) => fields.len(),
            };
            let read_struct_field = cx.ident_of("read_struct_field", trait_span);

            let path = cx.path_ident(trait_span, substr.type_ident);
            let result =
                decode_static_fields(cx, trait_span, path, summary, |cx, span, name, field| {
                    cx.expr_try(
                        span,
                        cx.expr_method_call(
                            span,
                            blkdecoder.clone(),
                            read_struct_field,
                            vec![
                                cx.expr_str(span, name),
                                cx.expr_usize(span, field),
                                exprdecode.clone(),
                            ],
                        ),
                    )
                });
            let result = cx.expr_ok(trait_span, result);
            cx.expr_method_call(
                trait_span,
                decoder,
                cx.ident_of("read_struct", trait_span),
                vec![
                    cx.expr_str(trait_span, substr.type_ident.name),
                    cx.expr_usize(trait_span, nfields),
                    cx.lambda1(trait_span, result, blkarg),
                ],
            )
        }
        StaticEnum(_, ref fields) => {
            let variant = cx.ident_of("i", trait_span);

            let mut arms = Vec::with_capacity(fields.len() + 1);
            let mut variants = Vec::with_capacity(fields.len());
            let rvariant_arg = cx.ident_of("read_enum_variant_arg", trait_span);

            for (i, &(ident, v_span, ref parts)) in fields.iter().enumerate() {
                variants.push(cx.expr_str(v_span, ident.name));

                let path = cx.path(trait_span, vec![substr.type_ident, ident]);
                let decoded =
                    decode_static_fields(cx, v_span, path, parts, |cx, span, _, field| {
                        let idx = cx.expr_usize(span, field);
                        cx.expr_try(
                            span,
                            cx.expr_method_call(
                                span,
                                blkdecoder.clone(),
                                rvariant_arg,
                                vec![idx, exprdecode.clone()],
                            ),
                        )
                    });

                arms.push(cx.arm(v_span, cx.pat_lit(v_span, cx.expr_usize(v_span, i)), decoded));
            }

            arms.push(cx.arm_unreachable(trait_span));

            let result = cx.expr_ok(
                trait_span,
                cx.expr_match(trait_span, cx.expr_ident(trait_span, variant), arms),
            );
            let lambda = cx.lambda(trait_span, vec![blkarg, variant], result);
            let variant_vec = cx.expr_vec(trait_span, variants);
            let variant_vec = cx.expr_addr_of(trait_span, variant_vec);
            let result = cx.expr_method_call(
                trait_span,
                blkdecoder,
                cx.ident_of("read_enum_variant", trait_span),
                vec![variant_vec, lambda],
            );
            cx.expr_method_call(
                trait_span,
                decoder,
                cx.ident_of("read_enum", trait_span),
                vec![
                    cx.expr_str(trait_span, substr.type_ident.name),
                    cx.lambda1(trait_span, result, blkarg),
                ],
            )
        }
        _ => cx.bug("expected StaticEnum or StaticStruct in derive(Decodable)"),
    }
}

/// Creates a decoder for a single enum variant/struct:
/// - `outer_pat_path` is the path to this enum variant/struct
/// - `getarg` should retrieve the `usize`-th field with name `@str`.
fn decode_static_fields<F>(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    outer_pat_path: ast::Path,
    fields: &StaticFields,
    mut getarg: F,
) -> P<Expr>
where
    F: FnMut(&mut ExtCtxt<'_>, Span, Symbol, usize) -> P<Expr>,
{
    match *fields {
        Unnamed(ref fields, is_tuple) => {
            let path_expr = cx.expr_path(outer_pat_path);
            if !is_tuple {
                path_expr
            } else {
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(i, &span)| getarg(cx, span, Symbol::intern(&format!("_field{}", i)), i))
                    .collect();

                cx.expr_call(trait_span, path_expr, fields)
            }
        }
        Named(ref fields) => {
            // use the field's span to get nicer error messages.
            let fields = fields
                .iter()
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
