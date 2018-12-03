// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic, proc_macro_span, proc_macro_def_site)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Span, Diagnostic};

fn parse(input: TokenStream) -> Result<(), Diagnostic> {
    let mut hi_spans = vec![];
    for tree in input {
        if let TokenTree::Ident(ref ident) = tree {
            if ident.to_string() == "hi" {
                hi_spans.push(ident.span());
            }
        }
    }

    if !hi_spans.is_empty() {
        return Err(Span::def_site()
                       .error("hello to you, too!")
                       .span_note(hi_spans, "found these 'hi's"));
    }

    Ok(())
}

#[proc_macro]
pub fn hello(input: TokenStream) -> TokenStream {
    if let Err(diag) = parse(input) {
        diag.emit();
    }

    TokenStream::new()
}
