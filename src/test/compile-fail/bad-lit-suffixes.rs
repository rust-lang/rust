// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate
    "foo"suffix //~ ERROR extern crate name with a suffix is illegal
     as foo;

extern
    "C"suffix //~ ERROR ABI spec with a suffix is illegal
    fn foo() {}

extern
    "C"suffix //~ ERROR ABI spec with a suffix is illegal
{}

fn main() {
    ""suffix; //~ ERROR str literal with a suffix is illegal
    b""suffix; //~ ERROR binary str literal with a suffix is illegal
    r#""#suffix; //~ ERROR str literal with a suffix is illegal
    br#""#suffix; //~ ERROR binary str literal with a suffix is illegal
    'a'suffix; //~ ERROR char literal with a suffix is illegal
    b'a'suffix; //~ ERROR byte literal with a suffix is illegal

    1234u1024; //~ ERROR illegal width `1024` for integer literal
    1234i1024; //~ ERROR illegal width `1024` for integer literal
    1234f1024; //~ ERROR illegal width `1024` for float literal
    1234.5f1024; //~ ERROR illegal width `1024` for float literal

    1234suffix; //~ ERROR illegal suffix `suffix` for numeric literal
    0b101suffix; //~ ERROR illegal suffix `suffix` for numeric literal
    1.0suffix; //~ ERROR illegal suffix `suffix` for float literal
    1.0e10suffix; //~ ERROR illegal suffix `suffix` for float literal
}
