// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Defines the crate attribute syntax for macro re-export.

use ast;
use attr::AttrMetaMethods;
use diagnostic::SpanHandler;

/// Return a vector of the names of all macros re-exported from the crate.
pub fn gather(diag: &SpanHandler, krate: &ast::Crate) -> Vec<String> {
    let usage = "malformed macro_reexport attribute, expected \
                 #![macro_reexport(ident, ident, ...)]";

    let mut reexported: Vec<String> = vec!();
    for attr in krate.attrs.iter() {
        if !attr.check_name("macro_reexport") {
            continue;
        }

        match attr.meta_item_list() {
            None => diag.span_err(attr.span, usage),
            Some(list) => for mi in list.iter() {
                match mi.node {
                    ast::MetaWord(ref word)
                        => reexported.push(word.to_string()),
                    _ => diag.span_err(mi.span, usage),
                }
            }
        }
    }

    reexported
}
