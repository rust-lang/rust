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
use symbol::{Symbol, keywords};
use syntax_pos::{DUMMY_SP, Span};
use codemap::{self, ExpnInfo, NameAndSpan, MacroAttribute};
use parse::ParseSess;
use ptr::P;

/// Craft a span that will be ignored by the stability lint's
/// call to codemap's is_internal check.
/// The expanded code uses the unstable `#[prelude_import]` attribute.
fn ignored_span(sess: &ParseSess, sp: Span) -> Span {
    let info = ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            format: MacroAttribute(Symbol::intern("std_inject")),
            span: None,
            allow_internal_unstable: true,
        }
    };
    let expn_id = sess.codemap().record_expansion(info);
    let mut sp = sp;
    sp.expn_id = expn_id;
    return sp;
}

pub fn injected_crate_name(krate: &ast::Crate) -> Option<&'static str> {
    if attr::contains_name(&krate.attrs, "no_core") {
        None
    } else if attr::contains_name(&krate.attrs, "no_std") {
        Some("core")
    } else {
        Some("std")
    }
}

pub fn maybe_inject_crates_ref(sess: &ParseSess,
                               mut krate: ast::Crate,
                               alt_std_name: Option<String>)
                               -> ast::Crate {
    let name = match injected_crate_name(&krate) {
        Some(name) => name,
        None => return krate,
    };

    let crate_name = Symbol::intern(&alt_std_name.unwrap_or(name.to_string()));

    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![attr::mk_attr_outer(attr::mk_attr_id(),
                                        attr::mk_word_item(Symbol::intern("macro_use")))],
        vis: ast::Visibility::Inherited,
        node: ast::ItemKind::ExternCrate(Some(crate_name)),
        ident: ast::Ident::from_str(name),
        id: ast::DUMMY_NODE_ID,
        span: DUMMY_SP,
    }));

    let span = ignored_span(sess, DUMMY_SP);
    krate.module.items.insert(0, P(ast::Item {
        attrs: vec![ast::Attribute {
            style: ast::AttrStyle::Outer,
            value: ast::MetaItem {
                name: Symbol::intern("prelude_import"),
                node: ast::MetaItemKind::Word,
                span: span,
            },
            id: attr::mk_attr_id(),
            is_sugared_doc: false,
            span: span,
        }],
        vis: ast::Visibility::Inherited,
        node: ast::ItemKind::Use(P(codemap::dummy_spanned(ast::ViewPathGlob(ast::Path {
            segments: ["{{root}}", name, "prelude", "v1"].into_iter().map(|name| {
                ast::Ident::from_str(name).into()
            }).collect(),
            span: span,
        })))),
        id: ast::DUMMY_NODE_ID,
        ident: keywords::Invalid.ident(),
        span: span,
    }));

    krate
}
