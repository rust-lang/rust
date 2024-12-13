use rustc_ast::attr::{AttributeExt, filter_by_name};
use rustc_session::Session;
use rustc_span::symbol::{Symbol, sym};

use crate::session_diagnostics;

pub fn allow_internal_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::allow_internal_unstable)
}

pub fn rustc_allow_const_fn_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::rustc_allow_const_fn_unstable)
}

fn allow_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
    symbol: Symbol,
) -> impl Iterator<Item = Symbol> + 'a {
    let attrs = filter_by_name(attrs, symbol);
    let list = attrs
        .filter_map(move |attr| {
            attr.meta_item_list().or_else(|| {
                sess.dcx().emit_err(session_diagnostics::ExpectsFeatureList {
                    span: attr.span(),
                    name: symbol.to_ident_string(),
                });
                None
            })
        })
        .flatten();

    list.into_iter().filter_map(move |it| {
        let name = it.ident().map(|ident| ident.name);
        if name.is_none() {
            sess.dcx().emit_err(session_diagnostics::ExpectsFeatures {
                span: it.span(),
                name: symbol.to_ident_string(),
            });
        }
        name
    })
}
