// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parsing utilities for writing procedural macros.

use syntax::parse::{ParseSess, filemap_to_tts};
use syntax::tokenstream::TokenStream;

/// Map a string to tts, using a made-up filename. For example, `lex("15")` will return a
/// TokenStream containing the literal 15.
pub fn lex(source_str: &str) -> TokenStream {
    let ps = ParseSess::new();
    TokenStream::from_tts(filemap_to_tts(&ps,
                                         ps.codemap().new_filemap("<procmacro_lex>".to_string(),
                                                                  None,
                                                                  source_str.to_owned())))
}
