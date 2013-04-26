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

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::mk_uniq_str;

pub fn expand_syntax_ext(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {

    let var = get_single_str_from_tts(cx, sp, tts, "env!");

    // FIXME (#2248): if this was more thorough it would manufacture an
    // Option<str> rather than just an maybe-empty string.

    let e = match os::getenv(var) {
      None => mk_uniq_str(cx, sp, ~""),
      Some(ref s) => mk_uniq_str(cx, sp, copy *s)
    };
    MRExpr(e)
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
