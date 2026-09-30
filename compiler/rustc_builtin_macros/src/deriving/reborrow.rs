use rustc_ast::{self as ast, AttrArgs, Generics, ItemKind, token};
use rustc_errors::E0802;
use rustc_expand::base::ExtCtxt;
use rustc_macros::Diagnostic;
use rustc_span::{Ident, Span, Symbol, sym};
use thin_vec::ThinVec;

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
    let Some(target) = coerce_shared_target(cx, span, item) else {
        return;
    };

    push_marker_impl(cx, span, ident, generics, sym::CoerceShared, vec![target], push);
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

fn coerce_shared_target(cx: &ExtCtxt<'_>, span: Span, item: &ast::Item) -> Option<Box<ast::Ty>> {
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
    let target = match parser.parse_ty() {
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

    Some(target)
}

fn push_marker_impl(
    cx: &ExtCtxt<'_>,
    span: Span,
    ident: Ident,
    generics: &Generics,
    trait_name: Symbol,
    trait_args: Vec<Box<ast::Ty>>,
    push: &mut dyn FnMut(Box<ast::Item>),
) {
    let trait_path = new_path(cx, span, &[sym::core, sym::marker, trait_name], trait_args);
    let trait_ref = cx.trait_ref(trait_path);

    let self_params: Vec<_> =
        generics.params.iter().map(|p| generic_param_to_arg(cx, p, p.span())).collect();
    let self_ty = cx.ty_path(cx.path_all(span, false, vec![ident], self_params));

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
