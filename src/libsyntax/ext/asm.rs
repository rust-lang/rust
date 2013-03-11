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
 * Inline assembly support.
 */

use core::prelude::*;

use ast;
use codemap::span;
use ext::base;
use ext::base::*;

pub fn expand_asm(cx: ext_ctxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let args = get_exprs_from_tts(cx, tts);
    if args.len() == 0 {
        cx.span_fatal(sp, "ast! takes at least 1 argument.");
    }
    let asm =
        expr_to_str(cx, args[0],
                    ~"inline assembly must be a string literal.");
    let cons = if args.len() > 1 {
        expr_to_str(cx, args[1],
                    ~"constraints must be a string literal.")
    } else { ~"" };

    MRExpr(@ast::expr {
        id: cx.next_id(),
        callee_id: cx.next_id(),
        node: ast::expr_inline_asm(@asm, @cons),
        span: sp
    })
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
