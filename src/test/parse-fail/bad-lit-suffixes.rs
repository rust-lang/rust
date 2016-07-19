// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error


extern
    "C"suffix //~ ERROR ABI spec with a suffix is invalid
    fn foo() {}

extern
    "C"suffix //~ ERROR ABI spec with a suffix is invalid
{}

fn main() {
    ""suffix; //~ ERROR string literal with a suffix is invalid
    b""suffix; //~ ERROR byte string literal with a suffix is invalid
    r#""#suffix; //~ ERROR string literal with a suffix is invalid
    br#""#suffix; //~ ERROR byte string literal with a suffix is invalid
    'a'suffix; //~ ERROR char literal with a suffix is invalid
    b'a'suffix; //~ ERROR byte literal with a suffix is invalid

    1234u1024; //~ ERROR invalid width `1024` for integer literal
    1234i1024; //~ ERROR invalid width `1024` for integer literal
    1234f1024; //~ ERROR invalid width `1024` for float literal
    1234.5f1024; //~ ERROR invalid width `1024` for float literal

    1234suffix; //~ ERROR invalid suffix `suffix` for numeric literal
    0b101suffix; //~ ERROR invalid suffix `suffix` for numeric literal
    1.0suffix; //~ ERROR invalid suffix `suffix` for float literal
    1.0e10suffix; //~ ERROR invalid suffix `suffix` for float literal
}
