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
use codemap;
use ext::base;
use print;

use std::rc::Rc;

pub fn expand_syntax_ext(cx: &mut base::ExtCtxt,
                         sp: codemap::Span,
                         tt: &[ast::TokenTree])
                      -> ~base::MacResult {

    cx.print_backtrace();
    println!("{}", print::pprust::tt_to_str(&ast::TTDelim(
                Rc::new(tt.iter().map(|x| (*x).clone()).collect()))));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any(sp)
}
