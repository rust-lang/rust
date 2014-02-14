// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use parse::token;

// SeqSep : a sequence separator (token)
// and whether a trailing separator is allowed.
pub struct SeqSep {
    sep: Option<token::Token>,
    trailing_sep_allowed: bool
}

pub fn seq_sep_trailing_disallowed(t: token::Token) -> SeqSep {
    SeqSep {
        sep: Some(t),
        trailing_sep_allowed: false,
    }
}
pub fn seq_sep_trailing_allowed(t: token::Token) -> SeqSep {
    SeqSep {
        sep: Some(t),
        trailing_sep_allowed: true,
    }
}
pub fn seq_sep_none() -> SeqSep {
    SeqSep {
        sep: None,
        trailing_sep_allowed: false,
    }
}
