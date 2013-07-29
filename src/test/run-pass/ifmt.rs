// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

struct A;
struct B;

#[fmt="foo"]
impl fmt::Signed for A {
    fn fmt(_: &A, f: &mut fmt::Formatter) { f.buf.write("aloha".as_bytes()); }
}
impl fmt::Signed for B {
    fn fmt(_: &B, f: &mut fmt::Formatter) { f.buf.write("adios".as_bytes()); }
}

pub fn main() {
    fn t(a: ~str, b: &str) { assert_eq!(a, b.to_owned()); }

    // Make sure there's a poly formatter that takes anything
    t(ifmt!("{}", 1), "1");
    t(ifmt!("{}", A), "{}");
    t(ifmt!("{}", ()), "()");
    t(ifmt!("{}", @(~1, "foo")), "@(~1, \"foo\")");

    // Various edge cases without formats
    t(ifmt!(""), "");
    t(ifmt!("hello"), "hello");
    t(ifmt!("hello \\{"), "hello {");

    // At least exercise all the formats
    t(ifmt!("{:b}", true), "true");
    t(ifmt!("{:c}", '☃'), "☃");
    t(ifmt!("{:d}", 10), "10");
    t(ifmt!("{:i}", 10), "10");
    t(ifmt!("{:u}", 10u), "10");
    t(ifmt!("{:o}", 10u), "12");
    t(ifmt!("{:x}", 10u), "a");
    t(ifmt!("{:X}", 10u), "A");
    t(ifmt!("{:s}", "foo"), "foo");
    t(ifmt!("{:p}", 0x1234 as *int), "0x1234");
    t(ifmt!("{:p}", 0x1234 as *mut int), "0x1234");
    t(ifmt!("{:d}", A), "aloha");
    t(ifmt!("{:d}", B), "adios");
    t(ifmt!("foo {:s} ☃☃☃☃☃☃", "bar"), "foo bar ☃☃☃☃☃☃");
    t(ifmt!("{1} {0}", 0, 1), "1 0");
    t(ifmt!("{foo} {bar}", foo=0, bar=1), "0 1");
    t(ifmt!("{foo} {1} {bar} {0}", 0, 1, foo=2, bar=3), "2 1 3 0");
    t(ifmt!("{} {0:s}", "a"), "a a");
    t(ifmt!("{} {0}", "a"), "\"a\" \"a\"");

    // Methods should probably work
    t(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 0u), "c0");
    t(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 1u), "a1");
    t(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 2u), "b2");
    t(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 3u), "d3");
    t(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "a"), "aa");
    t(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "b"), "bb");
    t(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "c"), "cc");
    t(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "d"), "dd");
    t(ifmt!("{1, select, a{#{0:s}} other{#{1}}}", "b", "a"), "ab");
    t(ifmt!("{1, select, a{#{0}} other{#{1}}}", "c", "b"), "bb");
}

