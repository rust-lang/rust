// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The compiler code necessary to support the env! extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */

use core::prelude::*;

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;

use core::os;

pub fn expand_syntax_ext(cx: @ExtCtxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {

    let var = get_single_str_from_tts(cx, sp, tts, "env!");

    // FIXME (#2248): if this was more thorough it would manufacture an
    // Option<str> rather than just an maybe-empty string.

    let e = match os::getenv(var) {
      None => cx.expr_str(sp, @""),
      Some(s) => cx.expr_str(sp, s.to_managed())
    };
    MRExpr(e)
}
