use rustc_ast::{self as ast, Generics, ItemKind, MetaItem, VariantData};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;
use rustc_span::symbol::{Ident, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_clone(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    // The simple form is `fn clone(&self) -> Self { *self }`, possibly with
    // some additional `AssertParamIsClone` assertions.
    //
    // We can use the simple form if either of the following are true.
    // - The type derives Copy and there are no generic parameters. (If we
    //   used the simple form with generics, we'd have to bound the generics
    //   with Clone + Copy, and then there'd be no Clone impl at all if the
    //   user fills in something that is Clone but not Copy. After
    //   specialization we can remove this no-generics limitation.)
    // - The item is a union. (Unions with generic parameters still can derive
    //   Clone because they require Copy for deriving, Clone alone is not
    //   enough. Whether Clone is implemented for fields is irrelevant so we
    //   don't assert it.)
    let bounds;
    let substructure;
    let is_simple;
    match item {
        Annotatable::Item(annitem) => match &annitem.kind {
            ItemKind::Struct(_, Generics { params, .. })
            | ItemKind::Enum(_, Generics { params, .. }) => {
                let container_id = cx.current_expansion.id.expn_data().parent.expect_local();
                let has_derive_copy = cx.resolver.has_derive_copy(container_id);
                if has_derive_copy
                    && !params
                        .iter()
                        .any(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }))
                {
                    bounds = vec![];
                    is_simple = true;
                    substructure = combine_substructure(Box::new(|c, s, sub| {
                        cs_clone_simple("Clone", c, s, sub, false)
                    }));
                } else {
                    bounds = vec![];
                    is_simple = false;
                    substructure =
                        combine_substructure(Box::new(|c, s, sub| cs_clone("Clone", c, s, sub)));
                }
            }
            ItemKind::Union(..) => {
                bounds = vec![Path(path_std!(marker::Copy))];
                is_simple = true;
                substructure = combine_substructure(Box::new(|c, s, sub| {
                    cs_clone_simple("Clone", c, s, sub, true)
                }));
            }
            _ => cx.dcx().span_bug(span, "`#[derive(Clone)]` on wrong item kind"),
        },

        _ => cx.dcx().span_bug(span, "`#[derive(Clone)]` on trait item or impl item"),
    }

    let trait_def = TraitDef {
        span,
        path: path_std!(clone::Clone),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: bounds,
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::clone,
            generics: Bounds::empty(),
            explicit_self: true,
            nonself_args: Vec::new(),
            ret_ty: Self_,
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: substructure,
        }],
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand_ext(cx, mitem, item, push, is_simple)
}

fn cs_clone_simple(
    name: &str,
    cx: &ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    is_union: bool,
) -> BlockOrExpr {
    let mut stmts = ThinVec::new();
    let mut seen_type_names = FxHashSet::default();
    let mut process_variant = |variant: &VariantData| {
        for field in variant.fields() {
            // This basic redundancy checking only prevents duplication of
            // assertions like `AssertParamIsClone<Foo>` where the type is a
            // simple name. That's enough to get a lot of cases, though.
            if let Some(name) = field.ty.kind.is_simple_path()
                && !seen_type_names.insert(name)
            {
                // Already produced an assertion for this type.
                // Anonymous structs or unions must be eliminated as they cannot be
                // type parameters.
            } else {
                // let _: AssertParamIsClone<FieldTy>;
                super::assert_ty_bounds(cx, &mut stmts, field.ty.clone(), field.span, &[
                    sym::clone,
                    sym::AssertParamIsClone,
                ]);
            }
        }
    };

    if is_union {
        // Just a single assertion for unions, that the union impls `Copy`.
        // let _: AssertParamIsCopy<Self>;
        let self_ty = cx.ty_path(cx.path_ident(trait_span, Ident::with_dummy_span(kw::SelfUpper)));
        super::assert_ty_bounds(cx, &mut stmts, self_ty, trait_span, &[
            sym::clone,
            sym::AssertParamIsCopy,
        ]);
    } else {
        match *substr.fields {
            StaticStruct(vdata, ..) => {
                process_variant(vdata);
            }
            StaticEnum(enum_def, ..) => {
                for variant in &enum_def.variants {
                    process_variant(&variant.data);
                }
            }
            _ => cx.dcx().span_bug(
                trait_span,
                format!("unexpected substructure in simple `derive({name})`"),
            ),
        }
    }
    BlockOrExpr::new_mixed(stmts, Some(cx.expr_deref(trait_span, cx.expr_self(trait_span))))
}

fn cs_clone(
    name: &str,
    cx: &ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> BlockOrExpr {
    let ctor_path;
    let all_fields;
    let fn_path = cx.std_path(&[sym::clone, sym::Clone, sym::clone]);
    let subcall = |cx: &ExtCtxt<'_>, field: &FieldInfo| {
        let args = thin_vec![field.self_expr.clone()];
        cx.expr_call_global(field.span, fn_path.clone(), args)
    };

    let vdata;
    match substr.fields {
        Struct(vdata_, af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident]);
            all_fields = af;
            vdata = *vdata_;
        }
        EnumMatching(.., variant, af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident, variant.ident]);
            all_fields = af;
            vdata = &variant.data;
        }
        EnumDiscr(..) | AllFieldlessEnum(..) => {
            cx.dcx().span_bug(trait_span, format!("enum discriminants in `derive({name})`",))
        }
        StaticEnum(..) | StaticStruct(..) => {
            cx.dcx().span_bug(trait_span, format!("associated function in `derive({name})`"))
        }
    }

    let expr = match *vdata {
        VariantData::Struct { .. } => {
            let fields = all_fields
                .iter()
                .map(|field| {
                    let Some(ident) = field.name else {
                        cx.dcx().span_bug(
                            trait_span,
                            format!("unnamed field in normal struct in `derive({name})`",),
                        );
                    };
                    let call = subcall(cx, field);
                    cx.field_imm(field.span, ident, call)
                })
                .collect::<ThinVec<_>>();

            cx.expr_struct(trait_span, ctor_path, fields)
        }
        VariantData::Tuple(..) => {
            let subcalls = all_fields.iter().map(|f| subcall(cx, f)).collect();
            let path = cx.expr_path(ctor_path);
            cx.expr_call(trait_span, path, subcalls)
        }
        VariantData::Unit(..) => cx.expr_path(ctor_path),
    };
    BlockOrExpr::new_expr(expr)
}
