use crate::ast;
use crate::attr;
use crate::edition::Edition;
use crate::ext::hygiene::{Mark, SyntaxContext};
use crate::symbol::{Symbol, keywords};
use crate::source_map::{ExpnInfo, MacroAttribute, dummy_spanned, hygiene, respan};
use crate::ptr::P;
use crate::tokenstream::TokenStream;

use std::cell::Cell;
use std::iter;
use syntax_pos::{DUMMY_SP, Span};

/// Craft a span that will be ignored by the stability lint's
/// call to source_map's `is_internal` check.
/// The expanded code uses the unstable `#[prelude_import]` attribute.
fn ignored_span(sp: Span) -> Span {
    let mark = Mark::fresh(Mark::root());
    mark.set_expn_info(ExpnInfo {
        call_site: DUMMY_SP,
        def_site: None,
        format: MacroAttribute(Symbol::intern("std_inject")),
        allow_internal_unstable: Some(vec![
            Symbol::intern("prelude_import"),
        ].into()),
        allow_internal_unsafe: false,
        local_inner_macros: false,
        edition: hygiene::default_edition(),
    });
    sp.with_ctxt(SyntaxContext::empty().apply_mark(mark))
}

pub fn injected_crate_name() -> Option<&'static str> {
    INJECTED_CRATE_NAME.with(|name| name.get())
}

thread_local! {
    static INJECTED_CRATE_NAME: Cell<Option<&'static str>> = Cell::new(None);
}

pub fn maybe_inject_crates_ref(
    mut krate: ast::Crate,
    alt_std_name: Option<&str>,
    edition: Edition,
) -> ast::Crate {
    let rust_2018 = edition >= Edition::Edition2018;

    // the first name in this list is the crate name of the crate with the prelude
    let names: &[&str] = if attr::contains_name(&krate.attrs, "no_core") {
        return krate;
    } else if attr::contains_name(&krate.attrs, "no_std") {
        if attr::contains_name(&krate.attrs, "compiler_builtins") {
            &["core"]
        } else {
            &["core", "compiler_builtins"]
        }
    } else {
        &["std"]
    };

    // .rev() to preserve ordering above in combination with insert(0, ...)
    let alt_std_name = alt_std_name.map(Symbol::intern);
    for orig_name in names.iter().rev() {
        let orig_name = Symbol::intern(orig_name);
        let mut rename = orig_name;
        // HACK(eddyb) gensym the injected crates on the Rust 2018 edition,
        // so they don't accidentally interfere with the new import paths.
        if rust_2018 {
            rename = orig_name.gensymed();
        }
        let orig_name = if rename != orig_name {
            Some(orig_name)
        } else {
            None
        };
        krate.module.items.insert(0, P(ast::Item {
            attrs: vec![attr::mk_attr_outer(DUMMY_SP,
                                            attr::mk_attr_id(),
                                            attr::mk_word_item(ast::Ident::from_str("macro_use")))],
            vis: dummy_spanned(ast::VisibilityKind::Inherited),
            node: ast::ItemKind::ExternCrate(alt_std_name.or(orig_name)),
            ident: ast::Ident::with_empty_ctxt(rename),
            id: ast::DUMMY_NODE_ID,
            span: DUMMY_SP,
            tokens: None,
        }));
    }

    // the crates have been injected, the assumption is that the first one is the one with
    // the prelude.
    let name = names[0];

    INJECTED_CRATE_NAME.with(|opt_name| opt_name.set(Some(name)));

    let span = ignored_span(DUMMY_SP);
    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![ast::Attribute {
            style: ast::AttrStyle::Outer,
            path: ast::Path::from_ident(ast::Ident::new(Symbol::intern("prelude_import"), span)),
            tokens: TokenStream::empty(),
            id: attr::mk_attr_id(),
            is_sugared_doc: false,
            span,
        }],
        vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
        node: ast::ItemKind::Use(P(ast::UseTree {
            prefix: ast::Path {
                segments: iter::once(keywords::PathRoot.ident())
                    .chain(
                        [name, "prelude", "v1"].iter().cloned()
                            .map(ast::Ident::from_str)
                    ).map(ast::PathSegment::from_ident).collect(),
                span,
            },
            kind: ast::UseTreeKind::Glob,
            span,
        })),
        id: ast::DUMMY_NODE_ID,
        ident: keywords::Invalid.ident(),
        span,
        tokens: None,
    }));

    krate
}
