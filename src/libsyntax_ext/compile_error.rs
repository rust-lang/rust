// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The compiler code necessary to support the compile_error! extension.

use syntax::ext::base::*;
use syntax::ext::base;
use syntax::feature_gate;
use syntax_pos::Span;
use syntax::tokenstream;

pub fn expand_compile_error<'cx>(cx: &'cx mut ExtCtxt,
                              sp: Span,
                              tts: &[tokenstream::TokenTree])
                              -> Box<base::MacResult + 'cx> {
    if !cx.ecfg.enable_compile_error() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       "compile_error",
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_COMPILE_ERROR);
        return DummyResult::expr(sp);
    }

    let var = match get_single_str_from_tts(cx, sp, tts, "compile_error!") {
        None => return DummyResult::expr(sp),
        Some(v) => v,
    };

    cx.span_err(sp, &var);

    DummyResult::any(sp)
}
