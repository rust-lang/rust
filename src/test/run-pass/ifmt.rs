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
    macro_rules! t(($a:expr, $b:expr) => { assert_eq!($a, $b.to_owned()) })

    // Make sure there's a poly formatter that takes anything
    t!(ifmt!("{:?}", 1), "1");
    t!(ifmt!("{:?}", A), "{}");
    t!(ifmt!("{:?}", ()), "()");
    t!(ifmt!("{:?}", @(~1, "foo")), "@(~1, \"foo\")");

    // Various edge cases without formats
    t!(ifmt!(""), "");
    t!(ifmt!("hello"), "hello");
    t!(ifmt!("hello \\{"), "hello {");

    // default formatters should work
    t!(ifmt!("{}", 1i), "1");
    t!(ifmt!("{}", 1i8), "1");
    t!(ifmt!("{}", 1i16), "1");
    t!(ifmt!("{}", 1i32), "1");
    t!(ifmt!("{}", 1i64), "1");
    t!(ifmt!("{}", 1u), "1");
    t!(ifmt!("{}", 1u8), "1");
    t!(ifmt!("{}", 1u16), "1");
    t!(ifmt!("{}", 1u32), "1");
    t!(ifmt!("{}", 1u64), "1");
    t!(ifmt!("{}", 1.0f), "1");
    t!(ifmt!("{}", 1.0f32), "1");
    t!(ifmt!("{}", 1.0f64), "1");
    t!(ifmt!("{}", "a"), "a");
    t!(ifmt!("{}", ~"a"), "a");
    t!(ifmt!("{}", @"a"), "a");
    t!(ifmt!("{}", false), "false");
    t!(ifmt!("{}", 'a'), "a");

    // At least exercise all the formats
    t!(ifmt!("{:b}", true), "true");
    t!(ifmt!("{:c}", '☃'), "☃");
    t!(ifmt!("{:d}", 10), "10");
    t!(ifmt!("{:i}", 10), "10");
    t!(ifmt!("{:u}", 10u), "10");
    t!(ifmt!("{:o}", 10u), "12");
    t!(ifmt!("{:x}", 10u), "a");
    t!(ifmt!("{:X}", 10u), "A");
    t!(ifmt!("{:s}", "foo"), "foo");
    t!(ifmt!("{:s}", ~"foo"), "foo");
    t!(ifmt!("{:s}", @"foo"), "foo");
    t!(ifmt!("{:p}", 0x1234 as *int), "0x1234");
    t!(ifmt!("{:p}", 0x1234 as *mut int), "0x1234");
    t!(ifmt!("{:d}", A), "aloha");
    t!(ifmt!("{:d}", B), "adios");
    t!(ifmt!("foo {:s} ☃☃☃☃☃☃", "bar"), "foo bar ☃☃☃☃☃☃");
    t!(ifmt!("{1} {0}", 0, 1), "1 0");
    t!(ifmt!("{foo} {bar}", foo=0, bar=1), "0 1");
    t!(ifmt!("{foo} {1} {bar} {0}", 0, 1, foo=2, bar=3), "2 1 3 0");
    t!(ifmt!("{} {0:s}", "a"), "a a");
    t!(ifmt!("{} {0}", "a"), "a a");

    // Methods should probably work
    t!(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 0u), "c0");
    t!(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 1u), "a1");
    t!(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 2u), "b2");
    t!(ifmt!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 3u), "d3");
    t!(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "a"), "aa");
    t!(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "b"), "bb");
    t!(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "c"), "cc");
    t!(ifmt!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "d"), "dd");
    t!(ifmt!("{1, select, a{#{0:s}} other{#{1}}}", "b", "a"), "ab");
    t!(ifmt!("{1, select, a{#{0}} other{#{1}}}", "c", "b"), "bb");

    // Formatting strings and their arguments
    t!(ifmt!("{:s}", "a"), "a");
    t!(ifmt!("{:4s}", "a"), "a   ");
    t!(ifmt!("{:>4s}", "a"), "   a");
    t!(ifmt!("{:<4s}", "a"), "a   ");
    t!(ifmt!("{:.4s}", "a"), "a");
    t!(ifmt!("{:4.4s}", "a"), "a   ");
    t!(ifmt!("{:4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(ifmt!("{:<4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(ifmt!("{:>4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(ifmt!("{:>10.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(ifmt!("{:2.4s}", "aaaaa"), "aaaa");
    t!(ifmt!("{:2.4s}", "aaaa"), "aaaa");
    t!(ifmt!("{:2.4s}", "aaa"), "aaa");
    t!(ifmt!("{:2.4s}", "aa"), "aa");
    t!(ifmt!("{:2.4s}", "a"), "a ");
    t!(ifmt!("{:0>2s}", "a"), "0a");
    t!(ifmt!("{:.*s}", 4, "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(ifmt!("{:.1$s}", "aaaaaaaaaaaaaaaaaa", 4), "aaaa");
    t!(ifmt!("{:1$s}", "a", 4), "a   ");
    t!(ifmt!("{:-#s}", "a"), "a");
    t!(ifmt!("{:+#s}", "a"), "a");

    // Formatting integers should select the right implementation based off the
    // type of the argument. Also, hex/octal/binary should be defined for
    // integers, but they shouldn't emit the negative sign.
    t!(ifmt!("{:d}", -1i), "-1");
    t!(ifmt!("{:d}", -1i8), "-1");
    t!(ifmt!("{:d}", -1i16), "-1");
    t!(ifmt!("{:d}", -1i32), "-1");
    t!(ifmt!("{:d}", -1i64), "-1");
    t!(ifmt!("{:t}", 1i), "1");
    t!(ifmt!("{:t}", 1i8), "1");
    t!(ifmt!("{:t}", 1i16), "1");
    t!(ifmt!("{:t}", 1i32), "1");
    t!(ifmt!("{:t}", 1i64), "1");
    t!(ifmt!("{:x}", 1i), "1");
    t!(ifmt!("{:x}", 1i8), "1");
    t!(ifmt!("{:x}", 1i16), "1");
    t!(ifmt!("{:x}", 1i32), "1");
    t!(ifmt!("{:x}", 1i64), "1");
    t!(ifmt!("{:X}", 1i), "1");
    t!(ifmt!("{:X}", 1i8), "1");
    t!(ifmt!("{:X}", 1i16), "1");
    t!(ifmt!("{:X}", 1i32), "1");
    t!(ifmt!("{:X}", 1i64), "1");
    t!(ifmt!("{:o}", 1i), "1");
    t!(ifmt!("{:o}", 1i8), "1");
    t!(ifmt!("{:o}", 1i16), "1");
    t!(ifmt!("{:o}", 1i32), "1");
    t!(ifmt!("{:o}", 1i64), "1");

    t!(ifmt!("{:u}", 1u), "1");
    t!(ifmt!("{:u}", 1u8), "1");
    t!(ifmt!("{:u}", 1u16), "1");
    t!(ifmt!("{:u}", 1u32), "1");
    t!(ifmt!("{:u}", 1u64), "1");
    t!(ifmt!("{:t}", 1u), "1");
    t!(ifmt!("{:t}", 1u8), "1");
    t!(ifmt!("{:t}", 1u16), "1");
    t!(ifmt!("{:t}", 1u32), "1");
    t!(ifmt!("{:t}", 1u64), "1");
    t!(ifmt!("{:x}", 1u), "1");
    t!(ifmt!("{:x}", 1u8), "1");
    t!(ifmt!("{:x}", 1u16), "1");
    t!(ifmt!("{:x}", 1u32), "1");
    t!(ifmt!("{:x}", 1u64), "1");
    t!(ifmt!("{:X}", 1u), "1");
    t!(ifmt!("{:X}", 1u8), "1");
    t!(ifmt!("{:X}", 1u16), "1");
    t!(ifmt!("{:X}", 1u32), "1");
    t!(ifmt!("{:X}", 1u64), "1");
    t!(ifmt!("{:o}", 1u), "1");
    t!(ifmt!("{:o}", 1u8), "1");
    t!(ifmt!("{:o}", 1u16), "1");
    t!(ifmt!("{:o}", 1u32), "1");
    t!(ifmt!("{:o}", 1u64), "1");

    // Test the flags for formatting integers
    t!(ifmt!("{:3d}", 1),  "  1");
    t!(ifmt!("{:>3d}", 1),  "  1");
    t!(ifmt!("{:>+3d}", 1), " +1");
    t!(ifmt!("{:<3d}", 1), "1  ");
    t!(ifmt!("{:#d}", 1), "1");
    t!(ifmt!("{:#x}", 10), "0xa");
    t!(ifmt!("{:#X}", 10), "0xA");
    t!(ifmt!("{:#5x}", 10), "  0xa");
    t!(ifmt!("{:#o}", 10), "0o12");
    t!(ifmt!("{:08x}", 10),  "0000000a");
    t!(ifmt!("{:8x}", 10),   "       a");
    t!(ifmt!("{:<8x}", 10),  "a       ");
    t!(ifmt!("{:>8x}", 10),  "       a");
    t!(ifmt!("{:#08x}", 10), "0x00000a");
    t!(ifmt!("{:08d}", -10), "-0000010");
    t!(ifmt!("{:x}", -1u8), "ff");
    t!(ifmt!("{:X}", -1u8), "FF");
    t!(ifmt!("{:t}", -1u8), "11111111");
    t!(ifmt!("{:o}", -1u8), "377");
    t!(ifmt!("{:#x}", -1u8), "0xff");
    t!(ifmt!("{:#X}", -1u8), "0xFF");
    t!(ifmt!("{:#t}", -1u8), "0b11111111");
    t!(ifmt!("{:#o}", -1u8), "0o377");

    // Signed combinations
    t!(ifmt!("{:+5d}", 1),  "   +1");
    t!(ifmt!("{:+5d}", -1), "   -1");
    t!(ifmt!("{:05d}", 1),   "00001");
    t!(ifmt!("{:05d}", -1),  "-0001");
    t!(ifmt!("{:+05d}", 1),  "+0001");
    t!(ifmt!("{:+05d}", -1), "-0001");

    // Some float stuff
    t!(ifmt!("{:f}", 1.0f), "1");
    t!(ifmt!("{:f}", 1.0f32), "1");
    t!(ifmt!("{:f}", 1.0f64), "1");
    t!(ifmt!("{:.3f}", 1.0f), "1.000");
    t!(ifmt!("{:10.3f}", 1.0f),   "     1.000");
    t!(ifmt!("{:+10.3f}", 1.0f),  "    +1.000");
    t!(ifmt!("{:+10.3f}", -1.0f), "    -1.000");
}

