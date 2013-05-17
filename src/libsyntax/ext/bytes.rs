// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* The compiler code necessary to support the bytes! extension. */

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::{mk_u8, mk_slice_vec_e};

pub fn expand_syntax_ext(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let var = get_single_str_from_tts(cx, sp, tts, "bytes!");
    let mut bytes = ~[];
    for var.each |byte| {
        bytes.push(mk_u8(cx, sp, byte));
    }
    let e = mk_slice_vec_e(cx, sp, bytes);
    MRExpr(e)
}
