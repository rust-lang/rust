use rustc_ast::{
    self as ast, AngleBracketedArg, AttrArgs, DUMMY_NODE_ID, GenericArg, GenericBound,
    GenericParam, GenericParamKind, Generics, ItemKind, WherePredicate, WherePredicateKind,
    WhereRegionPredicate, token,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::E0802;
use rustc_expand::base::ExtCtxt;
use rustc_macros::Diagnostic;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::new_path;

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

    let self_args: Vec<_> =
        generics.params.iter().map(|p| generic_param_to_arg(cx, p, p.span())).collect();

    push_marker_impl(cx, span, ident, generics, sym::Reborrow, Vec::new(), self_args, push);
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
    let self_args: Vec<_> =
        generics.params.iter().map(|p| generic_param_to_arg(cx, p, p.span())).collect();
    let Some((target, generics)) = coerce_shared_target(cx, span, item, generics) else {
        return;
    };

    push_marker_impl(cx, span, ident, &generics, sym::CoerceShared, vec![target], self_args, push);
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
    coerce_shared_item: &ast::Item,
    source_generics: &Generics,
) -> Option<(Box<ast::Ty>, Generics)> {
    let mut attrs =
        coerce_shared_item.attrs.iter().filter(|attr| attr.has_name(sym::coerce_shared));
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
        cx.dcx().emit_err(MalformedTargetType { span: target.span });
        return None;
    };

    let Some(last) = path.segments.last_mut() else {
        // It shouldn't be possible for segments to be empty.
        cx.dcx().emit_err(MalformedTargetType { span: path.span });
        return None;
    };

    let Some(rustc_ast::GenericArgs::AngleBracketed(target_args)) = last.args.as_deref_mut() else {
        // FIXME(reborrow): same as above.
        cx.dcx().emit_err(NoGenericsOnTargetType { span: last.span() });
        return None;
    };

    // Map to check generated lifetime arg names against.
    let mut lt_names: FxHashSet<Symbol> = target_args
        .args
        .iter()
        .filter_map(|arg| {
            if let AngleBracketedArg::Arg(GenericArg::Lifetime(arg)) = arg {
                Some(arg.ident.name)
            } else {
                None
            }
        })
        .collect();

    // struct Source<'a, 'b, T, const U> {} + coerce_shared(Target<'a>) =>
    // impl<'a, 'a_, T, const U> CoerceShared<Target<'a_>> for Source<'a, T, const U>
    //   where 'a: 'a_ {}
    let coerce_shared_trait_param_count = source_generics.params.len() + lt_names.len();
    let mut trait_params = ThinVec::with_capacity(coerce_shared_trait_param_count);
    let mut trait_where_clause = source_generics.where_clause.clone();

    // First add in the existing lifetimes.
    trait_params.extend(
        source_generics
            .params
            .iter()
            .take_while(|p| matches!(p.kind, rustc_ast::GenericParamKind::Lifetime))
            .cloned(),
    );

    // Remember how many lifetime parameters there were.
    let source_generics_lt_count = trait_params.len();

    // Replace all lifetime parameters 'a with a new 'a_ where 'a: 'a_ in the Target definition, and
    // push the new lifetimes into trait params.
    for arg in target_args.args.iter_mut() {
        let AngleBracketedArg::Arg(GenericArg::Lifetime(arg)) = arg else {
            continue;
        };

        // Eagerly intern the generated lifetime name - it is unlikely we have a naming conflict.
        let mut name_string = format!("{}_", arg.ident.as_str());
        let mut name = Symbol::intern(&name_string);
        while lt_names.contains(&name) {
            // Just keep piling on the underscores.
            name_string.push('_');
            name = Symbol::intern(&name_string);
        }
        // Now that we generated a unique lifetime name, add it into the set.
        lt_names.insert(name);

        // Create our lifetime, add it into the trait parameters and create a where-clause predicate
        // for it.
        let lt = rustc_ast::Lifetime { id: DUMMY_NODE_ID, ident: Ident::with_dummy_span(name) };
        trait_params.push(GenericParam {
            id: DUMMY_NODE_ID,
            ident: lt.ident,
            attrs: Default::default(),
            bounds: Default::default(),
            is_placeholder: false,
            kind: GenericParamKind::Lifetime,
            colon_span: None,
        });
        trait_where_clause.predicates.push(WherePredicate {
            attrs: Default::default(),
            kind: WherePredicateKind::RegionPredicate(WhereRegionPredicate {
                lifetime: *arg,
                bounds: thin_vec![GenericBound::Outlives(lt)],
            }),
            id: DUMMY_NODE_ID,
            span: DUMMY_SP,
            is_placeholder: false,
        });
        // Note: this mutates `target` from `Target<'a>` to `Target<'a_>`.
        *arg = lt;
    }

    // Finally add in any non-lifetime generics to the trait parameters: they must follow lifetimes
    // hence this ordering. This is also why we made note of lifetime parameter count.
    trait_params.extend_from_slice(&source_generics.params[source_generics_lt_count..]);

    Some((target, Generics { params: trait_params, where_clause: trait_where_clause, span }))
}

fn push_marker_impl(
    cx: &ExtCtxt<'_>,
    span: Span,
    ident: Ident,
    generics: &Generics,
    trait_name: Symbol,
    trait_args: Vec<Box<ast::Ty>>,
    self_args: Vec<GenericArg>,
    push: &mut dyn FnMut(Box<ast::Item>),
) {
    let trait_path = new_path(cx, span, &[sym::core, sym::marker, trait_name], trait_args);
    let trait_ref = cx.trait_ref(trait_path);

    let self_ty = cx.ty_path(cx.path_all(span, false, vec![ident], self_args));

    push(cx.item_trait_impl(
        span,
        thin_vec::thin_vec![cx.attr_word(sym::automatically_derived, span)],
        generics_without_defaults(generics),
        ast::Safety::Default,
        false,
        trait_ref,
        self_ty,
        ThinVec::new(),
    ));
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

#[derive(Diagnostic)]
#[diag("malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`", code = E0802)]
#[note("expected target type to be a user-defined type")]
struct MalformedTargetType {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`", code = E0802)]
#[note("expected target type to have generics")]
struct NoGenericsOnTargetType {
    #[primary_span]
    span: Span,
}
