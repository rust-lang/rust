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
use ast;
use codemap::{ExpnInfo, NameAndSpan, ExpnFormat};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use parse::parser::PathStyle;
use symbol::Symbol;
use syntax_pos::Span;

use std::collections::HashSet;

pub fn collect_derives(cx: &mut ExtCtxt, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if attr.path != "derive" {
            return true;
        }

        match attr.parse_list(cx.parse_sess,
                              |parser| parser.parse_path_allowing_meta(PathStyle::Mod)) {
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

pub fn add_derived_markers<T>(cx: &mut ExtCtxt, span: Span, traits: &[ast::Path], item: T) -> T
    where T: HasAttrs,
{
    let (mut names, mut pretty_name) = (HashSet::new(), "derive(".to_owned());
    for (i, path) in traits.iter().enumerate() {
        if i > 0 {
            pretty_name.push_str(", ");
        }
        pretty_name.push_str(&path.to_string());
        names.insert(unwrap_or!(path.segments.get(0), continue).identifier.name);
    }
    pretty_name.push(')');

    cx.current_expansion.mark.set_expn_info(ExpnInfo {
        call_site: span,
        callee: NameAndSpan {
            format: ExpnFormat::MacroAttribute(Symbol::intern(&pretty_name)),
            span: None,
            allow_internal_unstable: true,
        },
    });

    let span = Span { ctxt: cx.backtrace(), ..span };
    item.map_attrs(|mut attrs| {
        if names.contains(&Symbol::intern("Eq")) && names.contains(&Symbol::intern("PartialEq")) {
            let meta = cx.meta_word(span, Symbol::intern("structural_match"));
            attrs.push(cx.attribute(span, meta));
        }
        if names.contains(&Symbol::intern("Copy")) && names.contains(&Symbol::intern("Clone")) {
            let meta = cx.meta_word(span, Symbol::intern("rustc_copy_clone_marker"));
            attrs.push(cx.attribute(span, meta));
        }
        attrs
    })
}
