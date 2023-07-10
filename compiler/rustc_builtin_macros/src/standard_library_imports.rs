use rustc_ast::{self as ast, attr};
use rustc_expand::base::{ExtCtxt, ResolverExpand};
use rustc_expand::expand::ExpansionConfig;
use rustc_session::Session;
use rustc_span::edition::Edition::*;
use rustc_span::hygiene::AstPass;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::DUMMY_SP;
use thin_vec::thin_vec;

pub fn inject(
    krate: &mut ast::Crate,
    pre_configured_attrs: &[ast::Attribute],
    resolver: &mut dyn ResolverExpand,
    sess: &Session,
) -> usize {
    let orig_num_items = krate.items.len();
    let edition = sess.parse_sess.edition;

    // the first name in this list is the crate name of the crate with the prelude
    let names: &[Symbol] = if attr::contains_name(pre_configured_attrs, sym::no_core) {
        return 0;
    } else if attr::contains_name(pre_configured_attrs, sym::no_std) {
        if attr::contains_name(pre_configured_attrs, sym::compiler_builtins) {
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
        let ident_span = if edition >= Edition2018 { span } else { call_site };
        let item = if name == sym::compiler_builtins {
            // compiler_builtins is a private implementation detail. We only
            // need to insert it into the crate graph for linking and should not
            // expose any of its public API.
            //
            // FIXME(#113634) We should inject this during post-processing like
            // we do for the panic runtime, profiler runtime, etc.
            cx.item(
                span,
                Ident::new(kw::Underscore, ident_span),
                thin_vec![],
                ast::ItemKind::ExternCrate(Some(name)),
            )
        } else {
            cx.item(
                span,
                Ident::new(name, ident_span),
                thin_vec![cx.attr_word(sym::macro_use, span)],
                ast::ItemKind::ExternCrate(None),
            )
        };
        krate.items.insert(0, item);
    }

    // The crates have been injected, the assumption is that the first one is
    // the one with the prelude.
    let name = names[0];

    let root = (edition == Edition2015).then_some(kw::PathRoot);

    let import_path = root
        .iter()
        .chain(&[name, sym::prelude])
        .chain(&[match edition {
            Edition2015 => sym::rust_2015,
            Edition2018 => sym::rust_2018,
            Edition2021 => sym::rust_2021,
            Edition2024 => sym::rust_2024,
        }])
        .map(|&symbol| Ident::new(symbol, span))
        .collect();

    let use_item = cx.item(
        span,
        Ident::empty(),
        thin_vec![cx.attr_word(sym::prelude_import, span)],
        ast::ItemKind::Use(ast::UseTree {
            prefix: cx.path(span, import_path),
            kind: ast::UseTreeKind::Glob,
            span,
        }),
    );

    krate.items.insert(0, use_item);
    krate.items.len() - orig_num_items
}
