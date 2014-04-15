// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Deprecated fmt! syntax extension

use ast;
use codemap::Span;
use ext::base;
use ext::build::AstBuilder;

pub fn expand_syntax_ext(ecx: &mut base::ExtCtxt, sp: Span,
                         _tts: &[ast::TokenTree]) -> ~base::MacResult {
    ecx.span_err(sp, "`fmt!` is deprecated, use `format!` instead");
    ecx.parse_sess.span_diagnostic.span_note(sp,
        "see http://static.rust-lang.org/doc/master/std/fmt/index.html \
         for documentation");

    base::MacExpr::new(ecx.expr_uint(sp, 2))
}
