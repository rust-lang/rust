// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(macro_rules)]
#![deny(warnings)]
#![allow(unused_must_use)]
#![allow(deprecated_owned_vector)]

use std::fmt;
use std::io::MemWriter;
use std::io;
use std::str;

struct A;
struct B;

impl fmt::Signed for A {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.buf.write("aloha".as_bytes())
    }
}
impl fmt::Signed for B {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.buf.write("adios".as_bytes())
    }
}

macro_rules! t(($a:expr, $b:expr) => { assert_eq!($a, $b.to_owned()) })

pub fn main() {
    // Make sure there's a poly formatter that takes anything
    t!(format!("{:?}", 1), "1");
    t!(format!("{:?}", A), "A");
    t!(format!("{:?}", ()), "()");
    t!(format!("{:?}", @(~1, "foo")), "@(~1, \"foo\")");

    // Various edge cases without formats
    t!(format!(""), "");
    t!(format!("hello"), "hello");
    t!(format!("hello \\{"), "hello {");

    // default formatters should work
    t!(format!("{}", 1.0f32), "1");
    t!(format!("{}", 1.0f64), "1");
    t!(format!("{}", "a"), "a");
    t!(format!("{}", ~"a"), "a");
    t!(format!("{}", false), "false");
    t!(format!("{}", 'a'), "a");

    // At least exercise all the formats
    t!(format!("{:b}", true), "true");
    t!(format!("{:c}", '☃'), "☃");
    t!(format!("{:d}", 10), "10");
    t!(format!("{:i}", 10), "10");
    t!(format!("{:u}", 10u), "10");
    t!(format!("{:o}", 10u), "12");
    t!(format!("{:x}", 10u), "a");
    t!(format!("{:X}", 10u), "A");
    t!(format!("{:s}", "foo"), "foo");
    t!(format!("{:s}", ~"foo"), "foo");
    t!(format!("{:p}", 0x1234 as *int), "0x1234");
    t!(format!("{:p}", 0x1234 as *mut int), "0x1234");
    t!(format!("{:d}", A), "aloha");
    t!(format!("{:d}", B), "adios");
    t!(format!("foo {:s} ☃☃☃☃☃☃", "bar"), "foo bar ☃☃☃☃☃☃");
    t!(format!("{1} {0}", 0, 1), "1 0");
    t!(format!("{foo} {bar}", foo=0, bar=1), "0 1");
    t!(format!("{foo} {1} {bar} {0}", 0, 1, foo=2, bar=3), "2 1 3 0");
    t!(format!("{} {0}", "a"), "a a");
    t!(format!("{foo_bar}", foo_bar=1), "1");

    // Methods should probably work
    t!(format!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 0u), "c0");
    t!(format!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 1u), "a1");
    t!(format!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 2u), "b2");
    t!(format!("{0, plural, =1{a#} =2{b#} zero{c#} other{d#}}", 3u), "d3");
    t!(format!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "a"), "aa");
    t!(format!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "b"), "bb");
    t!(format!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "c"), "cc");
    t!(format!("{0, select, a{a#} b{b#} c{c#} other{d#}}", "d"), "dd");
    t!(format!("{1, select, a{#{0:s}} other{#}}", "b", "a"), "ab");
    t!(format!("{1, select, a{#{0}} other{#}}", "c", "b"), "b");

    // Formatting strings and their arguments
    t!(format!("{:s}", "a"), "a");
    t!(format!("{:4s}", "a"), "a   ");
    t!(format!("{:>4s}", "a"), "   a");
    t!(format!("{:<4s}", "a"), "a   ");
    t!(format!("{:.4s}", "a"), "a");
    t!(format!("{:4.4s}", "a"), "a   ");
    t!(format!("{:4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:<4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:>4.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:>10.4s}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:2.4s}", "aaaaa"), "aaaa");
    t!(format!("{:2.4s}", "aaaa"), "aaaa");
    t!(format!("{:2.4s}", "aaa"), "aaa");
    t!(format!("{:2.4s}", "aa"), "aa");
    t!(format!("{:2.4s}", "a"), "a ");
    t!(format!("{:0>2s}", "a"), "0a");
    t!(format!("{:.*s}", 4, "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:.1$s}", "aaaaaaaaaaaaaaaaaa", 4), "aaaa");
    t!(format!("{:.a$s}", "aaaaaaaaaaaaaaaaaa", a=4), "aaaa");
    t!(format!("{:1$s}", "a", 4), "a   ");
    t!(format!("{1:0$s}", 4, "a"), "a   ");
    t!(format!("{:a$s}", "a", a=4), "a   ");
    t!(format!("{:-#s}", "a"), "a");
    t!(format!("{:+#s}", "a"), "a");

    // Some float stuff
    t!(format!("{:f}", 1.0f32), "1");
    t!(format!("{:f}", 1.0f64), "1");
    t!(format!("{:.3f}", 1.0f64), "1.000");
    t!(format!("{:10.3f}", 1.0f64),   "     1.000");
    t!(format!("{:+10.3f}", 1.0f64),  "    +1.000");
    t!(format!("{:+10.3f}", -1.0f64), "    -1.000");

    t!(format!("{:e}", 1.2345e6f32), "1.2345e6");
    t!(format!("{:e}", 1.2345e6f64), "1.2345e6");
    t!(format!("{:E}", 1.2345e6f64), "1.2345E6");
    t!(format!("{:.3e}", 1.2345e6f64), "1.234e6");
    t!(format!("{:10.3e}", 1.2345e6f64),   "   1.234e6");
    t!(format!("{:+10.3e}", 1.2345e6f64),  "  +1.234e6");
    t!(format!("{:+10.3e}", -1.2345e6f64), "  -1.234e6");

    // Escaping
    t!(format!("\\{"), "{");
    t!(format!("\\}"), "}");
    t!(format!("\\#"), "#");
    t!(format!("\\\\"), "\\");

    test_write();
    test_print();
    test_order();

    // make sure that format! doesn't move out of local variables
    let a = ~3;
    format!("{:?}", a);
    format!("{:?}", a);

    // make sure that format! doesn't cause spurious unused-unsafe warnings when
    // it's inside of an outer unsafe block
    unsafe {
        let a: int = ::std::cast::transmute(3u);
        format!("{}", a);
    }

    test_format_args();

    // test that trailing commas are acceptable
    format!("{}", "test",);
    format!("{foo}", foo="test",);
}

// Basic test to make sure that we can invoke the `write!` macro with an
// io::Writer instance.
fn test_write() {
    let mut buf = MemWriter::new();
    write!(&mut buf as &mut io::Writer, "{}", 3);
    {
        let w = &mut buf as &mut io::Writer;
        write!(w, "{foo}", foo=4);
        write!(w, "{:s}", "hello");
        writeln!(w, "{}", "line");
        writeln!(w, "{foo}", foo="bar");
    }

    let s = str::from_utf8(buf.unwrap().as_slice()).unwrap().to_owned();
    t!(s, "34helloline\nbar\n");
}

// Just make sure that the macros are defined, there's not really a lot that we
// can do with them just yet (to test the output)
fn test_print() {
    print!("hi");
    print!("{:?}", vec!(0u8));
    println!("hello");
    println!("this is a {}", "test");
    println!("{foo}", foo="bar");
}

// Just make sure that the macros are defined, there's not really a lot that we
// can do with them just yet (to test the output)
fn test_format_args() {
    let mut buf = MemWriter::new();
    {
        let w = &mut buf as &mut io::Writer;
        format_args!(|args| { fmt::write(w, args); }, "{}", 1);
        format_args!(|args| { fmt::write(w, args); }, "test");
        format_args!(|args| { fmt::write(w, args); }, "{test}", test=3);
    }
    let s = str::from_utf8(buf.unwrap().as_slice()).unwrap().to_owned();
    t!(s, "1test3");

    let s = format_args!(fmt::format, "hello {}", "world");
    t!(s, "hello world");
}

fn test_order() {
    // Make sure format!() arguments are always evaluated in a left-to-right
    // ordering
    fn foo() -> int {
        static mut FOO: int = 0;
        unsafe {
            FOO += 1;
            FOO
        }
    }
    assert_eq!(format!("{} {} {a} {b} {} {c}",
                       foo(), foo(), foo(), a=foo(), b=foo(), c=foo()),
               ~"1 2 4 5 3 6");
}
