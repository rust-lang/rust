use ast::{AttrStyle, Ident, MacArgs, Path};
use rustc_ast::{ast, tokenstream::TokenStream};
use rustc_attr::{mk_attr, HasAttrs};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_feature::BUILTIN_ATTRIBUTE_MAP;
use rustc_parse::validate_attr;
use rustc_span::{sym, Span};

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    // Validate input against the `#[rustc_must_use]` template.
    let (_, _, template, _) = &BUILTIN_ATTRIBUTE_MAP[&sym::rustc_must_use];
    let attr = ecx.attribute(meta.clone());
    validate_attr::check_builtin_attribute(ecx.parse_sess, &attr, sym::must_use, template.clone());

    let reason = meta.name_value_literal();
    let mac_args = match reason {
        None => MacArgs::Empty,
        Some(lit) => MacArgs::Eq(span, TokenStream::new(vec![lit.token_tree().into()])),
    };

    // def-site context makes rustc accept the unstable `rustc_must_use`
    let span = ecx.with_def_site_ctxt(item.span());
    item.visit_attrs(|attrs| {
        attrs.push(mk_attr(
            AttrStyle::Outer,
            Path::from_ident(Ident::with_dummy_span(sym::rustc_must_use)),
            mac_args,
            span,
        ));
    });
    vec![item]
}
