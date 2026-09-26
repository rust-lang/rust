use rustc_ast::{
    self as ast, AngleBracketedArg, AttrArgs, DUMMY_NODE_ID, GenericArg, GenericBound,
    GenericParam, GenericParamKind, Generics, ItemKind, WherePredicate, WherePredicateKind,
    WhereRegionPredicate, token,
};
use rustc_errors::E0802;
use rustc_expand::base::ExtCtxt;
use rustc_macros::Diagnostic;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

macro_rules! path {
    ($span:expr, $($part:ident)::*) => { vec![$(Ident::new(sym::$part, $span),)*] }
}

pub(crate) fn expand_deriving_reborrow(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    _is_const: bool,
) {
    let Some((ident, generics)) = struct_def(cx, span, item, sym::Reborrow) else {
        return;
    };

    push_marker_impl(cx, span, ident, generics, sym::Reborrow, Vec::new(), push);
}

pub(crate) fn expand_deriving_coerce_shared(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    _is_const: bool,
) {
    let Some((ident, generics)) = struct_def(cx, span, item, sym::CoerceShared) else {
        return;
    };
    let Some((target, generics)) = coerce_shared_target(cx, span, item, generics) else {
        return;
    };

    push_marker_impl(
        cx,
        span,
        ident,
        &generics,
        sym::CoerceShared,
        vec![GenericArg::Type(target)],
        push,
    );
}

fn struct_def<'a>(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &'a ast::Item,
    trait_name: Symbol,
) -> Option<(Ident, &'a Generics)> {
    match &item.kind {
        ItemKind::Struct(ident, generics, _) => Some((*ident, generics)),
        ItemKind::Enum(..) => {
            cx.dcx().emit_err(UnsupportedItem { span, trait_name, kind: "enum" });
            None
        }
        ItemKind::Union(..) => {
            cx.dcx().emit_err(UnsupportedItem { span, trait_name, kind: "union" });
            None
        }
        _ => {
            cx.dcx().emit_err(UnsupportedItem { span, trait_name, kind: "item" });
            None
        }
    }
}

fn coerce_shared_target(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    generics: &Generics,
) -> Option<(Box<ast::Ty>, Generics)> {
    let mut attrs = item.attrs.iter().filter(|attr| attr.has_name(sym::coerce_shared));
    let Some(attr) = attrs.next() else {
        cx.dcx().emit_err(MissingTarget { span });
        return None;
    };
    if let Some(duplicate) = attrs.next() {
        cx.dcx().emit_err(DuplicateTarget { first: attr.span, duplicate: duplicate.span });
        return None;
    }

    let AttrArgs::Delimited(args) = &attr.get_normal_item().args else {
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    };
    if args.delim != token::Delimiter::Parenthesis || args.tokens.is_empty() {
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    }

    let mut parser = cx.new_parser_from_tts(args.tokens.clone());
    let mut target = match parser.parse_ty() {
        Ok(target) => target,
        Err(err) => {
            err.cancel();
            cx.dcx().emit_err(MalformedTarget { span: attr.span });
            return None;
        }
    };
    if parser.token != token::Eof {
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    }

    let rustc_ast::TyKind::Path(_, path) = &mut target.kind else {
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    };

    let Some(last) = path.segments.last_mut() else {
        // FIXME(reborrow): we might want to support CoerceShared<Foo> for Bar at some point.
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    };

    let Some(args) = last.args.as_deref_mut() else {
        // FIXME(reborrow): same as above.
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    };

    let rustc_ast::GenericArgs::AngleBracketed(args) = args else {
        cx.dcx().emit_err(MalformedTarget { span: attr.span });
        return None;
    };

    let mut own_params = generics.params.clone();
    let mut param_insert_index = own_params
        .iter()
        .enumerate()
        .find(|(_, arg)| {
            matches!(
                arg.kind,
                rustc_ast::GenericParamKind::Lifetime | rustc_ast::GenericParamKind::Type { .. }
            )
        })
        .map(|(i, _)| i)
        .unwrap_or(own_params.len());
    let mut own_where_clause = generics.where_clause.clone();

    for arg in args.args.iter_mut() {
        // Replace all lifetime parameters 'a with a new 'a_ where 'a: 'a_.
        let AngleBracketedArg::Arg(GenericArg::Lifetime(arg)) = arg else {
            continue;
        };
        let lt = rustc_ast::Lifetime {
            id: DUMMY_NODE_ID,
            ident: Ident::with_dummy_span(Symbol::intern(&format!("{}_", arg.ident.as_str()))),
        };
        // FIXME(reborrow): this is terribly inefficient.
        own_params.insert(
            param_insert_index,
            GenericParam {
                id: DUMMY_NODE_ID,
                ident: lt.ident,
                attrs: Default::default(),
                bounds: Default::default(),
                is_placeholder: false,
                kind: GenericParamKind::Lifetime,
                colon_span: None,
            },
        );
        param_insert_index += 1;
        own_where_clause.predicates.push(WherePredicate {
            attrs: Default::default(),
            kind: WherePredicateKind::RegionPredicate(WhereRegionPredicate {
                lifetime: *arg,
                bounds: thin_vec![GenericBound::Outlives(lt)],
            }),
            id: DUMMY_NODE_ID,
            span: DUMMY_SP,
            is_placeholder: false,
        });
        *arg = lt;
    }

    Some((target, Generics { params: own_params, where_clause: own_where_clause, span }))
}

fn push_marker_impl(
    cx: &ExtCtxt<'_>,
    span: Span,
    ident: Ident,
    generics: &Generics,
    trait_name: Symbol,
    trait_args: Vec<GenericArg>,
    push: &mut dyn FnMut(Box<ast::Item>),
) {
    let mut trait_parts = path!(span, core::marker);
    trait_parts.push(Ident::new(trait_name, span));
    let trait_path = cx.path_all(span, true, trait_parts, trait_args);
    let trait_ref = cx.trait_ref(trait_path);

    let self_params: Vec<_> = generics
        .params
        .iter()
        .filter_map(|param| match param.kind {
            GenericParamKind::Lifetime => {
                if trait_name == sym::CoerceShared && !param.ident.as_str().ends_with("_") {
                    None
                } else {
                    Some(GenericArg::Lifetime(cx.lifetime(param.span(), param.ident)))
                }
            }
            GenericParamKind::Type { .. } => {
                Some(GenericArg::Type(cx.ty_ident(param.span(), param.ident)))
            }
            GenericParamKind::Const { .. } => {
                Some(GenericArg::Const(cx.const_ident(param.span(), param.ident)))
            }
        })
        .collect();
    let self_ty = cx.ty_path(cx.path_all(span, false, vec![ident], self_params));

    push(cx.item(
        span,
        thin_vec::thin_vec![cx.attr_word(sym::automatically_derived, span)],
        ast::ItemKind::Impl(ast::Impl {
            generics: impl_generics(cx, generics),
            of_trait: Some(Box::new(ast::TraitImplHeader {
                safety: ast::Safety::Default,
                polarity: ast::ImplPolarity::Positive,
                defaultness: ast::Defaultness::Implicit,
                trait_ref,
            })),
            constness: ast::Const::No,
            self_ty,
            items: ThinVec::new(),
        }),
    ));
}

fn impl_generics(cx: &ExtCtxt<'_>, generics: &Generics) -> Generics {
    // Rebuild the generic parameter declarations because defaults are allowed on structs but
    // rejected on impls. Preserve lifetime, type, and const parameters and their bounds, const
    // parameter types, and the where-clause, while omitting type and const defaults.
    Generics {
        params: generics
            .params
            .iter()
            .map(|param| match &param.kind {
                GenericParamKind::Lifetime => {
                    cx.lifetime_param(param.span(), param.ident, param.bounds.clone())
                }
                GenericParamKind::Type { default: _ } => {
                    cx.typaram(param.span(), param.ident, param.bounds.clone(), None)
                }
                GenericParamKind::Const { ty, span: _, default: _ } => cx.const_param(
                    param.span(),
                    param.ident,
                    param.bounds.clone(),
                    ty.clone(),
                    None,
                ),
            })
            .collect(),
        where_clause: generics.where_clause.clone(),
        span: generics.span,
    }
}

#[derive(Diagnostic)]
#[diag("`derive({$trait_name})` is only supported for structs, not {$kind}s", code = E0802)]
struct UnsupportedItem {
    #[primary_span]
    span: Span,
    trait_name: Symbol,
    kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("`derive(CoerceShared)` requires exactly one `#[coerce_shared(Target)]` attribute", code = E0802)]
struct MissingTarget {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("duplicate `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`", code = E0802)]
struct DuplicateTarget {
    #[primary_span]
    duplicate: Span,
    #[note("first `#[coerce_shared(Target)]` attribute is here")]
    first: Span,
}

#[derive(Diagnostic)]
#[diag("malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`", code = E0802)]
#[note("expected a single target type, for example `#[coerce_shared(Target<'a, T>)]`")]
struct MalformedTarget {
    #[primary_span]
    span: Span,
}
