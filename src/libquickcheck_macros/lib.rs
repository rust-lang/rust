// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate provides the `#[quickcheck]` attribute. Its use is
//! documented in the `quickcheck` crate.

#![crate_id = "quickcheck_macros#0.11-pre"]
#![crate_type = "dylib"]
#![experimental]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![feature(macro_registrar, managed_boxes)]

extern crate syntax;

use syntax::ast;
use syntax::codemap;
use syntax::parse::token;
use syntax::ext::base::{ SyntaxExtension, ItemModifier, ExtCtxt };
use syntax::ext::build::AstBuilder;

/// For the `#[quickcheck]` attribute. Do not use.
#[macro_registrar]
#[doc(hidden)]
pub fn macro_registrar(register: |ast::Name, SyntaxExtension|) {
    register(token::intern("quickcheck"), ItemModifier(expand_meta_quickcheck));
}

/// Expands the `#[quickcheck]` attribute.
///
/// Expands:
/// ```
/// #[quickcheck]
/// fn check_something(_: uint) -> bool {
///     true
/// }
/// ```
/// to:
/// ```
/// #[test]
/// fn check_something() {
///     fn check_something(_: uint) -> bool {
///         true
///     }
///     ::quickcheck::quickcheck(check_something)
/// }
/// ```
fn expand_meta_quickcheck(cx: &mut ExtCtxt,
                          span: codemap::Span,
                          _: @ast::MetaItem,
                          item: @ast::Item) -> @ast::Item {
    match item.node {
        ast::ItemFn(..) | ast::ItemStatic(..) => {
            // Copy original function without attributes
            let prop = @ast::Item {attrs: Vec::new(), ..(*item).clone()};
            // ::quickcheck::quickcheck
            let check_ident = token::str_to_ident("quickcheck");
            let check_path = vec!(check_ident, check_ident);
            // Wrap original function in new outer function, calling ::quickcheck::quickcheck()
            let fn_decl = @codemap::respan(span, ast::DeclItem(prop));
            let inner_fn = @codemap::respan(span, ast::StmtDecl(fn_decl, ast::DUMMY_NODE_ID));
            let inner_ident = cx.expr_ident(span, prop.ident);
            let check_call = cx.expr_call_global(span, check_path, vec![inner_ident]);
            let body = cx.block(span, vec![inner_fn], Some(check_call));
            let test = cx.item_fn(span, item.ident, Vec::new(), cx.ty_nil(), body);

            // Copy attributes from original function
            let mut attrs = item.attrs.clone();
            // Add #[test] attribute
            attrs.push(cx.attribute(span, cx.meta_word(span, token::intern_and_get_ident("test"))));
            // Attach the attributes to the outer function
            @ast::Item {attrs: attrs, ..(*test).clone()}
        },
        _ => {
            cx.span_err(span, "#[quickcheck] only supported on statics and functions");
            item
        }
    }
}
