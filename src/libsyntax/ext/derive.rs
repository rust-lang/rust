// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr::HasAttrs;
use {ast, codemap};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use parse::parser::PathStyle;
use symbol::Symbol;
use syntax_pos::Span;

pub fn collect_derives(cx: &mut ExtCtxt, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if attr.path != "derive" {
            return true;
        }

        match attr.parse_list(cx.parse_sess, |parser| parser.parse_path(PathStyle::Mod)) {
            Ok(ref traits) if traits.is_empty() => {
                cx.span_warn(attr.span, "empty trait list in `derive`");
                false
            }
            Ok(traits) => {
                result.extend(traits);
                true
            }
            Err(mut e) => {
                e.emit();
                false
            }
        }
    });
    result
}

fn allow_unstable(cx: &mut ExtCtxt, span: Span, attr_name: &str) -> Span {
    Span {
        expn_id: cx.codemap().record_expansion(codemap::ExpnInfo {
            call_site: span,
            callee: codemap::NameAndSpan {
                format: codemap::MacroAttribute(Symbol::intern(attr_name)),
                span: Some(span),
                allow_internal_unstable: true,
            },
        }),
        ..span
    }
}

pub fn add_derived_markers<T: HasAttrs>(cx: &mut ExtCtxt, traits: &[ast::Path], item: T) -> T {
    let span = match traits.get(0) {
        Some(path) => path.span,
        None => return item,
    };

    item.map_attrs(|mut attrs| {
        if traits.iter().any(|path| *path == "PartialEq") &&
           traits.iter().any(|path| *path == "Eq") {
            let span = allow_unstable(cx, span, "derive(PartialEq, Eq)");
            let meta = cx.meta_word(span, Symbol::intern("structural_match"));
            attrs.push(cx.attribute(span, meta));
        }
        if traits.iter().any(|path| *path == "Copy") &&
           traits.iter().any(|path| *path == "Clone") {
            let span = allow_unstable(cx, span, "derive(Copy, Clone)");
            let meta = cx.meta_word(span, Symbol::intern("rustc_copy_clone_marker"));
            attrs.push(cx.attribute(span, meta));
        }
        attrs
    })
}
