use rustc_ast::ast::DUMMY_NODE_ID;
use rustc_ast::attr::mk_list_item;
use rustc_ast::ptr::P;
use rustc_ast::{
    AngleBracketedArg, AngleBracketedArgs, Arm, AssocItem, AssocItemKind, BinOpKind, BindingMode,
    Block, Const, Defaultness, EnumDef, Expr, Fn, FnHeader, FnRetTy, FnSig, GenericArg,
    GenericArgs, Generics, Impl, ImplPolarity, Item, ItemKind, LitIntType, LitKind, MetaItem,
    Mutability, NestedMetaItem, PatKind, Path, PathSegment, Stmt, StmtKind, Ty, Unsafe, Variant,
    VariantData, Visibility, VisibilityKind,
};
use rustc_attr::{find_repr_attrs, int_type_of_word};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::struct_span_err;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, Symbol, DUMMY_SP};

pub fn expand_from_repr(
    cx: &mut ExtCtxt<'_>,
    derive_span: Span,
    _mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    if let Some(ctx) = Ctx::extract(cx, item, derive_span) {
        push(make_from_repr(cx, ctx.enum_def, ctx.ty, ctx.repr_ty));
    }
}

struct Ctx<'enumdef> {
    enum_def: &'enumdef EnumDef,
    ty: P<Ty>,
    repr_ty: P<Ty>,
}

impl<'enumdef> Ctx<'enumdef> {
    fn extract(
        cx: &mut ExtCtxt<'_>,
        item: &'enumdef Annotatable,
        derive_span: Span,
    ) -> Option<Ctx<'enumdef>> {
        match *item {
            Annotatable::Item(ref annitem) => match annitem.kind {
                ItemKind::Enum(ref enum_def, _) => {
                    let repr = extract_repr(cx, annitem, enum_def, derive_span)?;

                    let ty = cx.ty_ident(DUMMY_SP, annitem.ident);
                    let repr_ty = cx.ty_ident(DUMMY_SP, Ident { name: repr, span: DUMMY_SP });

                    return Some(Self { enum_def, ty, repr_ty });
                }
                _ => {}
            },
            _ => {}
        }
        struct_span_err!(
            &cx.sess.parse_sess.span_diagnostic,
            derive_span,
            E0789,
            "`FromRepr` can only be derived for enums",
        )
        .span_label(item.span(), "not an enum")
        .emit();
        None
    }
}

fn extract_repr(
    cx: &mut ExtCtxt<'_>,
    annitem: &P<Item>,
    def: &EnumDef,
    derive_span: Span,
) -> Option<Symbol> {
    let reprs: Vec<_> = annitem
        .attrs
        .iter()
        .flat_map(|attr| find_repr_attrs(&cx.sess, attr))
        .filter_map(|r| {
            use rustc_attr::*;
            match r {
                ReprInt(rustc_attr::IntType::UnsignedInt(int_type)) => Some(int_type.name()),
                ReprInt(rustc_attr::IntType::SignedInt(int_type)) => Some(int_type.name()),
                ReprC | ReprPacked(..) | ReprSimd | ReprTransparent | ReprAlign(..)
                | ReprNoNiche => None,
            }
        })
        .collect();

    if reprs.len() != 1 {
        let mut err_spans = Vec::new();

        if reprs.is_empty() {
            err_spans.push(derive_span);
        } else {
            for attr in &annitem.attrs {
                if attr.has_name(sym::repr) {
                    err_spans.push(attr.span);
                }
            }
        }

        let mut err = struct_span_err!(
            &cx.sess.parse_sess.span_diagnostic,
            err_spans,
            E0789,
            "`FromRepr` can only be derived for enums with exactly one explicit integer representation",
        );
        if reprs.is_empty() {
            err.span_label(annitem.span, "enum missing numeric `repr` annotation");
        }
        let too_many_reprs = reprs.len() > 1;
        for (i, attr) in annitem.attrs.iter().enumerate() {
            if attr.has_name(sym::repr) {
                if too_many_reprs {
                    if i == 0 {
                        err.span_label(
                            attr.span,
                            "multiple `repr` annotations on the same item are not allowed",
                        );
                    } else {
                        err.span_label(attr.span, "");
                    }
                }
                if let Some(items) = attr.meta_item_list() {
                    for item in items {
                        if int_type_of_word(item.name_or_empty()).is_none() {
                            err.span_label(attr.span, "Non-integer representation");
                        }
                    }
                }
            }
        }
        err.emit();
        return None;
    }

    if !def.is_c_like() {
        let mut err = struct_span_err!(
            &cx.sess.parse_sess.span_diagnostic,
            derive_span,
            E0789,
            "`FromRepr` can only be derived for C-like enums",
        );
        err.span_label(derive_span, "derive from `FromRepr` trait");
        for variant in &def.variants {
            match variant.data {
                VariantData::Struct(..) | VariantData::Tuple(..) => {
                    err.span_label(variant.span, "variant is not data-free");
                }
                VariantData::Unit(..) => {}
            }
        }
        err.note("only unit variants, like `Foo,` or `Bar = 3,` are data-free");
        err.emit();
        return None;
    }
    return Some(reprs[0]);
}

fn make_from_repr(cx: &mut ExtCtxt<'_>, def: &EnumDef, ty: P<Ty>, repr_ty: P<Ty>) -> Annotatable {
    let param_ident = Ident::from_str("value");
    let result_type = make_result_type(cx, ty.clone(), repr_ty.clone());

    let try_from_repr = {
        let decl = cx.fn_decl(
            vec![cx.param(DUMMY_SP, param_ident.clone(), repr_ty.clone())],
            FnRetTy::Ty(result_type),
        );

        assoc_item(
            "try_from_repr",
            AssocItemKind::Fn(Box::new(Fn {
                defaultness: Defaultness::Final,
                generics: Generics::default(),
                sig: FnSig { header: FnHeader::default(), decl, span: DUMMY_SP },
                body: Some(make_match_block(cx, def, repr_ty.clone(), param_ident)),
            })),
        )
    };

    Annotatable::Item(cx.item(
        DUMMY_SP,
        Ident::empty(),
        vec![],
        ItemKind::Impl(Box::new(Impl {
            unsafety: Unsafe::Yes(DUMMY_SP),
            polarity: ImplPolarity::Positive,
            defaultness: Defaultness::Final,
            constness: Const::No,
            generics: Generics::default(),
            of_trait: Some(cx.trait_ref(std_path_from_ident_symbols(
                cx,
                &[Symbol::intern("enums"), sym::FromRepr],
            ))),
            self_ty: ty.clone(),
            items: vec![try_from_repr],
        })),
    ))
}

fn make_match_block(
    cx: &mut ExtCtxt<'_>,
    def: &EnumDef,
    repr_ty: P<Ty>,
    value_ident: Ident,
) -> P<Block> {
    let ok = std_path_from_ident_symbols(cx, &[sym::result, sym::Result, sym::Ok]);
    let ok = cx.expr_path(ok);

    let mut prev_explicit_disr: Option<P<Expr>> = None;
    let mut count_since_prev_explicit_disr = 0;

    let mut stmts = Vec::with_capacity(def.variants.len() + 1);
    let mut arms = Vec::with_capacity(def.variants.len() + 1);

    for Variant { ident, disr_expr, .. } in &def.variants {
        let expr = match (disr_expr, &prev_explicit_disr) {
            (Some(disr), _) => {
                prev_explicit_disr = Some(disr.value.clone());
                count_since_prev_explicit_disr = 0;
                disr.value.clone()
            }
            (None, None) => {
                let expr = cx.expr_lit(
                    DUMMY_SP,
                    LitKind::Int(count_since_prev_explicit_disr, LitIntType::Unsuffixed),
                );
                count_since_prev_explicit_disr += 1;
                expr
            }
            (None, Some(prev_expr)) => {
                count_since_prev_explicit_disr += 1;
                cx.expr_binary(
                    DUMMY_SP,
                    BinOpKind::Add,
                    prev_expr.clone(),
                    cx.expr_lit(
                        DUMMY_SP,
                        LitKind::Int(count_since_prev_explicit_disr, LitIntType::Unsuffixed),
                    ),
                )
            }
        };

        let const_ident = Ident::from_str(&format!("DISCIMINANT_FOR_{}", arms.len()));
        stmts.push(
            cx.stmt_item(DUMMY_SP, cx.item_const(DUMMY_SP, const_ident, repr_ty.clone(), expr)),
        );

        arms.push(cx.arm(
            DUMMY_SP,
            cx.pat_ident(DUMMY_SP, const_ident),
            cx.expr_call(
                DUMMY_SP,
                ok.clone(),
                vec![cx.expr_path(cx.path(DUMMY_SP, vec![Ident::from_str("Self"), ident.clone()]))],
            ),
        ));
    }

    let err = std_path_from_ident_symbols(cx, &[sym::result, sym::Result, sym::Err]);
    let try_from_int_error = std_path_from_ident_symbols(
        cx,
        &[Symbol::intern("enums"), Symbol::intern("TryFromReprError")],
    );

    let other_match = Ident::from_str("other_value");

    // Rather than having to know how many variants could fit into the repr,
    // just allow the catch-all to be superfluous.
    arms.push(Arm {
        attrs: ThinVec::from(vec![cx.attribute(mk_list_item(
            Ident::new(sym::allow, DUMMY_SP),
            vec![NestedMetaItem::MetaItem(
                cx.meta_word(DUMMY_SP, Symbol::intern("unreachable_patterns")),
            )],
        ))]),

        pat: cx.pat(
            DUMMY_SP,
            PatKind::Ident(BindingMode::ByValue(Mutability::Not), other_match.clone(), None),
        ),
        guard: Option::None,
        body: cx.expr_call(
            DUMMY_SP,
            cx.expr_path(err),
            vec![cx.expr_call(
                DUMMY_SP,
                cx.expr_path(try_from_int_error),
                vec![cx.expr_ident(DUMMY_SP, other_match.clone())],
            )],
        ),
        span: DUMMY_SP,
        id: DUMMY_NODE_ID,
        is_placeholder: false,
    });

    stmts.push(Stmt {
        id: DUMMY_NODE_ID,
        span: DUMMY_SP,
        kind: StmtKind::Expr(cx.expr_match(DUMMY_SP, cx.expr_ident(DUMMY_SP, value_ident), arms)),
    });

    cx.block(DUMMY_SP, stmts)
}

fn assoc_item(name: &str, kind: AssocItemKind) -> P<AssocItem> {
    P(AssocItem {
        attrs: vec![],
        id: DUMMY_NODE_ID,
        span: DUMMY_SP,
        vis: Visibility { kind: VisibilityKind::Inherited, span: DUMMY_SP, tokens: None },
        ident: Ident::from_str(name),
        kind,
        tokens: None,
    })
}

fn make_result_type(cx: &mut ExtCtxt<'_>, ty: P<Ty>, repr_ty: P<Ty>) -> P<Ty> {
    std_path_with_generics(cx, &[sym::result], sym::Result, vec![ty, error_type(cx, repr_ty)])
}

fn std_path_with_generics(
    cx: &ExtCtxt<'_>,
    symbols: &[Symbol],
    ty: Symbol,
    generics: Vec<P<Ty>>,
) -> P<Ty> {
    let mut path = std_path_from_ident_symbols(cx, symbols);
    path.segments.push(path_segment_with_generics(ty, generics));
    cx.ty_path(path)
}

fn std_path_from_ident_symbols(cx: &ExtCtxt<'_>, symbols: &[Symbol]) -> Path {
    Path {
        span: DUMMY_SP,
        segments: cx.std_path(symbols).into_iter().map(PathSegment::from_ident).collect(),
        tokens: None,
    }
}

fn error_type(cx: &ExtCtxt<'_>, repr_ty: P<Ty>) -> P<Ty> {
    let mut error_type = std_path_from_ident_symbols(cx, &[Symbol::intern("enums")]);

    error_type
        .segments
        .push(path_segment_with_generics(Symbol::intern("TryFromReprError"), vec![repr_ty]));

    cx.ty_path(error_type)
}

fn path_segment_with_generics(symbol: Symbol, generic_types: Vec<P<Ty>>) -> PathSegment {
    PathSegment {
        ident: Ident { name: symbol, span: DUMMY_SP },
        id: DUMMY_NODE_ID,
        args: Some(P(GenericArgs::AngleBracketed(AngleBracketedArgs {
            span: DUMMY_SP,
            args: generic_types
                .into_iter()
                .map(|ty| AngleBracketedArg::Arg(GenericArg::Type(ty)))
                .collect(),
        }))),
    }
}
