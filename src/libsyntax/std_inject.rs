// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use attr;
use std::cell::Cell;
use ext::hygiene::{Mark, SyntaxContext};
use symbol::{Symbol, keywords};
use syntax_pos::{DUMMY_SP, Span};
use codemap::{ExpnInfo, NameAndSpan, MacroAttribute};
use ptr::P;
use tokenstream::TokenStream;

/// Craft a span that will be ignored by the stability lint's
/// call to codemap's `is_internal` check.
/// The expanded code uses the unstable `#[prelude_import]` attribute.
fn ignored_span(sp: Span) -> Span {
    let mark = Mark::fresh(Mark::root());
    mark.set_expn_info(ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            format: MacroAttribute(Symbol::intern("std_inject")),
            span: None,
            allow_internal_unstable: true,
            allow_internal_unsafe: false,
        }
    });
    sp.with_ctxt(SyntaxContext::empty().apply_mark(mark))
}

pub fn injected_crate_name() -> Option<&'static str> {
    INJECTED_CRATE_NAME.with(|name| name.get())
}

thread_local! {
    static INJECTED_CRATE_NAME: Cell<Option<&'static str>> = Cell::new(None);
}

pub fn maybe_inject_crates_ref(mut krate: ast::Crate, alt_std_name: Option<String>) -> ast::Crate {
    let name = if attr::contains_name(&krate.attrs, "no_core") {
        return krate;
    } else if attr::contains_name(&krate.attrs, "no_std") {
        "core"
    } else {
        "std"
    };

    INJECTED_CRATE_NAME.with(|opt_name| opt_name.set(Some(name)));

    let crate_name = Symbol::intern(&alt_std_name.unwrap_or_else(|| name.to_string()));

    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![attr::mk_attr_outer(DUMMY_SP,
                                        attr::mk_attr_id(),
                                        attr::mk_word_item(Symbol::intern("macro_use")))],
        vis: ast::Visibility::Inherited,
        node: ast::ItemKind::ExternCrate(Some(crate_name)),
        ident: ast::Ident::from_str(name),
        id: ast::DUMMY_NODE_ID,
        span: DUMMY_SP,
        tokens: None,
    }));

    let span = ignored_span(DUMMY_SP);
    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![ast::Attribute {
            style: ast::AttrStyle::Outer,
            path: ast::Path::from_ident(span, ast::Ident::from_str("prelude_import")),
            tokens: TokenStream::empty(),
            id: attr::mk_attr_id(),
            is_sugared_doc: false,
            span,
        }],
        vis: ast::Visibility::Inherited,
        node: ast::ItemKind::Use(P(ast::UseTree {
            prefix: ast::Path {
                segments: ["{{root}}", name, "prelude", "v1"].into_iter().map(|name| {
                    ast::PathSegment::from_ident(ast::Ident::from_str(name), DUMMY_SP)
                }).collect(),
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
