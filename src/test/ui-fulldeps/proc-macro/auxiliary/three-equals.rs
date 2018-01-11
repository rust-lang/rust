// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
#![feature(proc_macro)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenNode, Span, Diagnostic};

fn parse(input: TokenStream) -> Result<(), Diagnostic> {
    let mut count = 0;
    let mut last_span = Span::def_site();
    for tree in input {
        let span = tree.span;
        if count >= 3 {
            return Err(span.error(format!("expected EOF, found `{}`.", tree))
                           .span_note(last_span, "last good input was here")
                           .help("input must be: `===`"))
        }

        if let TokenNode::Op('=', _) = tree.kind {
            count += 1;
        } else {
            return Err(span.error(format!("expected `=`, found `{}`.", tree)));
        }

        last_span = span;
    }

    if count < 3 {
        return Err(Span::def_site()
                       .error(format!("found {} equal signs, need exactly 3", count))
                       .help("input must be: `===`"))
    }

    Ok(())
}

#[proc_macro]
pub fn three_equals(input: TokenStream) -> TokenStream {
    if let Err(diag) = parse(input) {
        diag.emit();
        return TokenStream::empty();
    }

    "3".parse().unwrap()
}
