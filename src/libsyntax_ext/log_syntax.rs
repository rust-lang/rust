// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ext::base;
use syntax::feature_gate;
use syntax::print;
use syntax::tokenstream;
use syntax_pos;

pub fn expand_syntax_ext<'cx>(cx: &'cx mut base::ExtCtxt,
                              sp: syntax_pos::Span,
                              tts: &[tokenstream::TokenTree])
                              -> Box<base::MacResult + 'cx> {
    if !cx.ecfg.enable_log_syntax() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       "log_syntax",
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_LOG_SYNTAX);
        return base::DummyResult::any(sp);
    }

    println!("{}", print::pprust::tts_to_string(tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any(sp)
}
