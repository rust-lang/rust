use rustc_ast as ast;
use rustc_expand::base::{ExtCtxt, ResolverExpand};
use rustc_expand::expand::ExpansionConfig;
use rustc_session::Session;
use rustc_span::edition::Edition::*;
use rustc_span::hygiene::AstPass;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::DUMMY_SP;

pub fn inject(
    mut krate: ast::Crate,
    resolver: &mut dyn ResolverExpand,
    sess: &Session,
) -> ast::Crate {
    let edition = sess.parse_sess.edition;

    // the first name in this list is the crate name of the crate with the prelude
    let names: &[Symbol] = if sess.contains_name(&krate.attrs, sym::no_core) {
        return krate;
    } else if sess.contains_name(&krate.attrs, sym::no_std) {
        if sess.contains_name(&krate.attrs, sym::compiler_builtins) {
            &[sym::core]
        } else {
            &[sym::core, sym::compiler_builtins]
        }
    } else {
        &[sym::std]
    };

    let expn_id = resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::StdImports,
        &[sym::prelude_import],
        None,
    );
    let span = DUMMY_SP.with_def_site_ctxt(expn_id.to_expn_id());
    let call_site = DUMMY_SP.with_call_site_ctxt(expn_id.to_expn_id());

    let ecfg = ExpansionConfig::default("std_lib_injection".to_string());
    let cx = ExtCtxt::new(sess, ecfg, resolver, None);

    // .rev() to preserve ordering above in combination with insert(0, ...)
    for &name in names.iter().rev() {
        let ident = if edition >= Edition2018 {
            Ident::new(name, span)
        } else {
            Ident::new(name, call_site)
        };
        krate.items.insert(
            0,
            cx.item(
                span,
                ident,
                vec![cx.attribute(cx.meta_word(span, sym::macro_use))],
                ast::ItemKind::ExternCrate(None),
            ),
        );
    }

    // The crates have been injected, the assumption is that the first one is
    // the one with the prelude.
    let name = names[0];

    let root = (edition == Edition2015).then(|| kw::PathRoot);

    let import_path = root
        .iter()
        .chain(&[name, sym::prelude])
        .chain(&[match edition {
            Edition2015 => sym::rust_2015,
            Edition2018 => sym::rust_2018,
            Edition2021 => sym::rust_2021,
        }])
        .map(|&symbol| Ident::new(symbol, span))
        .collect();

    let use_item = cx.item(
        span,
        Ident::empty(),
        vec![cx.attribute(cx.meta_word(span, sym::prelude_import))],
        ast::ItemKind::Use(ast::UseTree {
            prefix: cx.path(span, import_path),
            kind: ast::UseTreeKind::Glob,
            span,
        }),
    );

    krate.items.insert(0, use_item);

    krate
}
