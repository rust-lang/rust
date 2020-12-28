use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, GenericArg, Generics, ItemKind, MetaItem, VariantData};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;

pub fn expand_deriving_clone(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    // check if we can use a short form
    //
    // the short form is `fn clone(&self) -> Self { *self }`
    //
    // we can use the short form if:
    // - the item is Copy (unfortunately, all we can check is whether it's also deriving Copy)
    // - there are no generic parameters (after specialization this limitation can be removed)
    //      if we used the short form with generics, we'd have to bound the generics with
    //      Clone + Copy, and then there'd be no Clone impl at all if the user fills in something
    //      that is Clone but not Copy. and until specialization we can't write both impls.
    // - the item is a union with Copy fields
    //      Unions with generic parameters still can derive Clone because they require Copy
    //      for deriving, Clone alone is not enough.
    //      Whever Clone is implemented for fields is irrelevant so we don't assert it.
    let bounds;
    let substructure;
    let is_shallow;
    match *item {
        Annotatable::Item(ref annitem) => match annitem.kind {
            ItemKind::Struct(_, Generics { ref params, .. })
            | ItemKind::Enum(_, Generics { ref params, .. }) => {
                let container_id = cx.current_expansion.id.expn_data().parent;
                if cx.resolver.has_derive_copy(container_id)
                    && !params
                        .iter()
                        .any(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }))
                {
                    bounds = vec![];
                    is_shallow = true;
                    substructure = combine_substructure(Box::new(|c, s, sub| {
                        cs_clone_shallow("Clone", c, s, sub, false)
                    }));
                } else {
                    bounds = vec![];
                    is_shallow = false;
                    substructure =
                        combine_substructure(Box::new(|c, s, sub| cs_clone("Clone", c, s, sub)));
                }
            }
            ItemKind::Union(..) => {
                bounds = vec![Literal(path_std!(marker::Copy))];
                is_shallow = true;
                substructure = combine_substructure(Box::new(|c, s, sub| {
                    cs_clone_shallow("Clone", c, s, sub, true)
                }));
            }
            _ => {
                bounds = vec![];
                is_shallow = false;
                substructure =
                    combine_substructure(Box::new(|c, s, sub| cs_clone("Clone", c, s, sub)));
            }
        },

        _ => cx.span_bug(span, "`#[derive(Clone)]` on trait item or impl item"),
    }

    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(inline)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(clone::Clone),
        additional_bounds: bounds,
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::clone,
            generics: Bounds::empty(),
            explicit_self: borrowed_explicit_self(),
            args: Vec::new(),
            ret_ty: Self_,
            attributes: attrs,
            is_unsafe: false,
            unify_fieldless_variants: false,
            combine_substructure: substructure,
        }],
        associated_types: Vec::new(),
    };

    trait_def.expand_ext(cx, mitem, item, push, is_shallow)
}

fn cs_clone_shallow(
    name: &str,
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    is_union: bool,
) -> P<Expr> {
    fn assert_ty_bounds(
        cx: &mut ExtCtxt<'_>,
        stmts: &mut Vec<ast::Stmt>,
        ty: P<ast::Ty>,
        span: Span,
        helper_name: &str,
    ) {
        // Generate statement `let _: helper_name<ty>;`,
        // set the expn ID so we can use the unstable struct.
        let span = cx.with_def_site_ctxt(span);
        let assert_path = cx.path_all(
            span,
            true,
            cx.std_path(&[sym::clone, Symbol::intern(helper_name)]),
            vec![GenericArg::Type(ty)],
        );
        stmts.push(cx.stmt_let_type_only(span, cx.ty_path(assert_path)));
    }
    fn process_variant(cx: &mut ExtCtxt<'_>, stmts: &mut Vec<ast::Stmt>, variant: &VariantData) {
        for field in variant.fields() {
            // let _: AssertParamIsClone<FieldTy>;
            assert_ty_bounds(cx, stmts, field.ty.clone(), field.span, "AssertParamIsClone");
        }
    }

    let mut stmts = Vec::new();
    if is_union {
        // let _: AssertParamIsCopy<Self>;
        let self_ty = cx.ty_path(cx.path_ident(trait_span, Ident::with_dummy_span(kw::SelfUpper)));
        assert_ty_bounds(cx, &mut stmts, self_ty, trait_span, "AssertParamIsCopy");
    } else {
        match *substr.fields {
            StaticStruct(vdata, ..) => {
                process_variant(cx, &mut stmts, vdata);
            }
            StaticEnum(enum_def, ..) => {
                for variant in &enum_def.variants {
                    process_variant(cx, &mut stmts, &variant.data);
                }
            }
            _ => cx.span_bug(
                trait_span,
                &format!(
                    "unexpected substructure in \
                                                    shallow `derive({})`",
                    name
                ),
            ),
        }
    }
    stmts.push(cx.stmt_expr(cx.expr_deref(trait_span, cx.expr_self(trait_span))));
    cx.expr_block(cx.block(trait_span, stmts))
}

fn cs_clone(
    name: &str,
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> P<Expr> {
    let ctor_path;
    let all_fields;
    let fn_path = cx.std_path(&[sym::clone, sym::Clone, sym::clone]);
    let subcall = |cx: &mut ExtCtxt<'_>, field: &FieldInfo<'_>| {
        let args = vec![cx.expr_addr_of(field.span, field.self_.clone())];
        cx.expr_call_global(field.span, fn_path.clone(), args)
    };

    let vdata;
    match *substr.fields {
        Struct(vdata_, ref af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident]);
            all_fields = af;
            vdata = vdata_;
        }
        EnumMatching(.., variant, ref af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident, variant.ident]);
            all_fields = af;
            vdata = &variant.data;
        }
        EnumNonMatchingCollapsed(..) => {
            cx.span_bug(trait_span, &format!("non-matching enum variants in `derive({})`", name,))
        }
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span, &format!("associated function in `derive({})`", name))
        }
    }

    match *vdata {
        VariantData::Struct(..) => {
            let fields = all_fields
                .iter()
                .map(|field| {
                    let ident = match field.name {
                        Some(i) => i,
                        None => cx.span_bug(
                            trait_span,
                            &format!("unnamed field in normal struct in `derive({})`", name,),
                        ),
                    };
                    let call = subcall(cx, field);
                    cx.field_imm(field.span, ident, call)
                })
                .collect::<Vec<_>>();

            cx.expr_struct(trait_span, ctor_path, fields)
        }
        VariantData::Tuple(..) => {
            let subcalls = all_fields.iter().map(|f| subcall(cx, f)).collect();
            let path = cx.expr_path(ctor_path);
            cx.expr_call(trait_span, path, subcalls)
        }
        VariantData::Unit(..) => cx.expr_path(ctor_path),
    }
}
