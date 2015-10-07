// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only
static c3: char =
    '\x1' //~ ERROR: numeric character escape is too short
;

static s: &'static str =
    "\x1" //~ ERROR: numeric character escape is too short
;

static c: char =
    '\●' //~ ERROR: unknown character escape
;

static s: &'static str =
    "\●" //~ ERROR: unknown character escape
;

// THIS MUST BE LAST, since it kills the lexer

static c: char =
    '●  //~ ERROR: character literal may only contain one codepoint
;
