use crate::ast;
use crate::attr;
use crate::edition::Edition;
use crate::ext::hygiene::{Mark, SyntaxContext};
use crate::symbol::{Ident, Symbol, kw, sym};
use crate::source_map::{ExpnInfo, MacroAttribute, dummy_spanned, respan};
use crate::ptr::P;
use crate::tokenstream::TokenStream;

use std::cell::Cell;
use std::iter;
use syntax_pos::{DUMMY_SP, Span};

/// Craft a span that will be ignored by the stability lint's
/// call to source_map's `is_internal` check.
/// The expanded code uses the unstable `#[prelude_import]` attribute.
fn ignored_span(sp: Span, edition: Edition) -> Span {
    let mark = Mark::fresh(Mark::root());
    mark.set_expn_info(ExpnInfo::with_unstable(
        MacroAttribute(Symbol::intern("std_inject")), sp, edition, &[sym::prelude_import]
    ));
    sp.with_ctxt(SyntaxContext::empty().apply_mark(mark))
}

pub fn injected_crate_name() -> Option<&'static str> {
    INJECTED_CRATE_NAME.with(|name| name.get())
}

thread_local! {
    // A `Symbol` might make more sense here, but it doesn't work, probably for
    // reasons relating to the use of thread-local storage for the Symbol
    // interner.
    static INJECTED_CRATE_NAME: Cell<Option<&'static str>> = Cell::new(None);
}

pub fn maybe_inject_crates_ref(
    mut krate: ast::Crate,
    alt_std_name: Option<&str>,
    edition: Edition,
) -> ast::Crate {
    let rust_2018 = edition >= Edition::Edition2018;

    // the first name in this list is the crate name of the crate with the prelude
    let names: &[&str] = if attr::contains_name(&krate.attrs, sym::no_core) {
        return krate;
    } else if attr::contains_name(&krate.attrs, sym::no_std) {
        if attr::contains_name(&krate.attrs, sym::compiler_builtins) {
            &["core"]
        } else {
            &["core", "compiler_builtins"]
        }
    } else {
        &["std"]
    };

    // .rev() to preserve ordering above in combination with insert(0, ...)
    let alt_std_name = alt_std_name.map(Symbol::intern);
    for orig_name_str in names.iter().rev() {
        // HACK(eddyb) gensym the injected crates on the Rust 2018 edition,
        // so they don't accidentally interfere with the new import paths.
        let orig_name_sym = Symbol::intern(orig_name_str);
        let orig_name_ident = Ident::with_empty_ctxt(orig_name_sym);
        let (rename, orig_name) = if rust_2018 {
            (orig_name_ident.gensym(), Some(orig_name_sym))
        } else {
            (orig_name_ident, None)
        };
        krate.module.items.insert(0, P(ast::Item {
            attrs: vec![attr::mk_attr_outer(
                DUMMY_SP,
                attr::mk_attr_id(),
                attr::mk_word_item(ast::Ident::with_empty_ctxt(sym::macro_use))
            )],
            vis: dummy_spanned(ast::VisibilityKind::Inherited),
            node: ast::ItemKind::ExternCrate(alt_std_name.or(orig_name)),
            ident: rename,
            id: ast::DUMMY_NODE_ID,
            span: DUMMY_SP,
            tokens: None,
        }));
    }

    // the crates have been injected, the assumption is that the first one is the one with
    // the prelude.
    let name = names[0];

    INJECTED_CRATE_NAME.with(|opt_name| opt_name.set(Some(name)));

    let span = ignored_span(DUMMY_SP, edition);
    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![ast::Attribute {
            style: ast::AttrStyle::Outer,
            path: ast::Path::from_ident(ast::Ident::new(sym::prelude_import, span)),
            tokens: TokenStream::empty(),
            id: attr::mk_attr_id(),
            is_sugared_doc: false,
            span,
        }],
        vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
        node: ast::ItemKind::Use(P(ast::UseTree {
            prefix: ast::Path {
                segments: iter::once(ast::Ident::with_empty_ctxt(kw::PathRoot))
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
        ident: ast::Ident::invalid(),
        span,
        tokens: None,
    }));

    krate
}
