#![deny(warnings)]
#![allow(unused_must_use)]
#![allow(unused_features)]
#![feature(box_syntax)]

use std::cell::RefCell;
use std::fmt::{self, Write};
use std::usize;

struct A;
struct B;
struct C;
struct D;

impl fmt::LowerHex for A {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("aloha")
    }
}
impl fmt::UpperHex for B {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("adios")
    }
}
impl fmt::Display for C {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "☃", "123")
    }
}
impl fmt::Binary for D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("aa")?;
        f.write_char('☃')?;
        f.write_str("bb")
    }
}

macro_rules! t {
    ($a:expr, $b:expr) => { assert_eq!($a, $b) }
}

pub fn main() {
    // Various edge cases without formats
    t!(format!(""), "");
    t!(format!("hello"), "hello");
    t!(format!("hello {{"), "hello {");

    // default formatters should work
    t!(format!("{}", 1.0f32), "1");
    t!(format!("{}", 1.0f64), "1");
    t!(format!("{}", "a"), "a");
    t!(format!("{}", "a".to_string()), "a");
    t!(format!("{}", false), "false");
    t!(format!("{}", 'a'), "a");

    // At least exercise all the formats
    t!(format!("{}", true), "true");
    t!(format!("{}", '☃'), "☃");
    t!(format!("{}", 10), "10");
    t!(format!("{}", 10_usize), "10");
    t!(format!("{:?}", '☃'), "'☃'");
    t!(format!("{:?}", 10), "10");
    t!(format!("{:?}", 10_usize), "10");
    t!(format!("{:?}", "true"), "\"true\"");
    t!(format!("{:?}", "foo\nbar"), "\"foo\\nbar\"");
    t!(format!("{:?}", "foo\n\"bar\"\r\n\'baz\'\t\\qux\\"),
       r#""foo\n\"bar\"\r\n\'baz\'\t\\qux\\""#);
    t!(format!("{:?}", "foo\0bar\x01baz\u{7f}q\u{75}x"),
       r#""foo\u{0}bar\u{1}baz\u{7f}qux""#);
    t!(format!("{:o}", 10_usize), "12");
    t!(format!("{:x}", 10_usize), "a");
    t!(format!("{:X}", 10_usize), "A");
    t!(format!("{}", "foo"), "foo");
    t!(format!("{}", "foo".to_string()), "foo");
    if cfg!(target_pointer_width = "32") {
        t!(format!("{:#p}", 0x1234 as *const isize), "0x00001234");
        t!(format!("{:#p}", 0x1234 as *mut isize), "0x00001234");
    } else {
        t!(format!("{:#p}", 0x1234 as *const isize), "0x0000000000001234");
        t!(format!("{:#p}", 0x1234 as *mut isize), "0x0000000000001234");
    }
    t!(format!("{:p}", 0x1234 as *const isize), "0x1234");
    t!(format!("{:p}", 0x1234 as *mut isize), "0x1234");
    t!(format!("{:x}", A), "aloha");
    t!(format!("{:X}", B), "adios");
    t!(format!("foo {} ☃☃☃☃☃☃", "bar"), "foo bar ☃☃☃☃☃☃");
    t!(format!("{1} {0}", 0, 1), "1 0");
    t!(format!("{foo} {bar}", foo=0, bar=1), "0 1");
    t!(format!("{foo} {1} {bar} {0}", 0, 1, foo=2, bar=3), "2 1 3 0");
    t!(format!("{} {0}", "a"), "a a");
    t!(format!("{foo_bar}", foo_bar=1), "1");
    t!(format!("{}", 5 + 5), "10");
    t!(format!("{:#4}", C), "☃123");
    t!(format!("{:b}", D), "aa☃bb");

    let a: &dyn fmt::Debug = &1;
    t!(format!("{:?}", a), "1");


    // Formatting strings and their arguments
    t!(format!("{}", "a"), "a");
    t!(format!("{:4}", "a"), "a   ");
    t!(format!("{:4}", "☃"), "☃   ");
    t!(format!("{:>4}", "a"), "   a");
    t!(format!("{:<4}", "a"), "a   ");
    t!(format!("{:^5}", "a"),  "  a  ");
    t!(format!("{:^5}", "aa"), " aa  ");
    t!(format!("{:^4}", "a"),  " a  ");
    t!(format!("{:^4}", "aa"), " aa ");
    t!(format!("{:.4}", "a"), "a");
    t!(format!("{:4.4}", "a"), "a   ");
    t!(format!("{:4.4}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:<4.4}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:>4.4}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:^4.4}", "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:>10.4}", "aaaaaaaaaaaaaaaaaa"), "      aaaa");
    t!(format!("{:2.4}", "aaaaa"), "aaaa");
    t!(format!("{:2.4}", "aaaa"), "aaaa");
    t!(format!("{:2.4}", "aaa"), "aaa");
    t!(format!("{:2.4}", "aa"), "aa");
    t!(format!("{:2.4}", "a"), "a ");
    t!(format!("{:0>2}", "a"), "0a");
    t!(format!("{:.*}", 4, "aaaaaaaaaaaaaaaaaa"), "aaaa");
    t!(format!("{:.1$}", "aaaaaaaaaaaaaaaaaa", 4), "aaaa");
    t!(format!("{:.a$}", "aaaaaaaaaaaaaaaaaa", a=4), "aaaa");
    t!(format!("{:1$}", "a", 4), "a   ");
    t!(format!("{1:0$}", 4, "a"), "a   ");
    t!(format!("{:a$}", "a", a=4), "a   ");
    t!(format!("{:-#}", "a"), "a");
    t!(format!("{:+#}", "a"), "a");
    t!(format!("{:/^10.8}", "1234567890"), "/12345678/");

    // Some float stuff
    t!(format!("{:}", 1.0f32), "1");
    t!(format!("{:}", 1.0f64), "1");
    t!(format!("{:.3}", 1.0f64), "1.000");
    t!(format!("{:10.3}", 1.0f64),   "     1.000");
    t!(format!("{:+10.3}", 1.0f64),  "    +1.000");
    t!(format!("{:+10.3}", -1.0f64), "    -1.000");

    t!(format!("{:e}", 1.2345e6f32), "1.2345e6");
    t!(format!("{:e}", 1.2345e6f64), "1.2345e6");
    t!(format!("{:E}", 1.2345e6f64), "1.2345E6");
    t!(format!("{:.3e}", 1.2345e6f64), "1.234e6");
    t!(format!("{:10.3e}", 1.2345e6f64),   "   1.234e6");
    t!(format!("{:+10.3e}", 1.2345e6f64),  "  +1.234e6");
    t!(format!("{:+10.3e}", -1.2345e6f64), "  -1.234e6");

    // Float edge cases
    t!(format!("{}", -0.0), "0");
    t!(format!("{:?}", -0.0), "-0.0");
    t!(format!("{:?}", 0.0), "0.0");

    // sign aware zero padding
    t!(format!("{:<3}", 1), "1  ");
    t!(format!("{:>3}", 1), "  1");
    t!(format!("{:^3}", 1), " 1 ");
    t!(format!("{:03}", 1), "001");
    t!(format!("{:<03}", 1), "001");
    t!(format!("{:>03}", 1), "001");
    t!(format!("{:^03}", 1), "001");
    t!(format!("{:+03}", 1), "+01");
    t!(format!("{:<+03}", 1), "+01");
    t!(format!("{:>+03}", 1), "+01");
    t!(format!("{:^+03}", 1), "+01");
    t!(format!("{:#05x}", 1), "0x001");
    t!(format!("{:<#05x}", 1), "0x001");
    t!(format!("{:>#05x}", 1), "0x001");
    t!(format!("{:^#05x}", 1), "0x001");
    t!(format!("{:05}", 1.2), "001.2");
    t!(format!("{:<05}", 1.2), "001.2");
    t!(format!("{:>05}", 1.2), "001.2");
    t!(format!("{:^05}", 1.2), "001.2");
    t!(format!("{:05}", -1.2), "-01.2");
    t!(format!("{:<05}", -1.2), "-01.2");
    t!(format!("{:>05}", -1.2), "-01.2");
    t!(format!("{:^05}", -1.2), "-01.2");
    t!(format!("{:+05}", 1.2), "+01.2");
    t!(format!("{:<+05}", 1.2), "+01.2");
    t!(format!("{:>+05}", 1.2), "+01.2");
    t!(format!("{:^+05}", 1.2), "+01.2");

    // Ergonomic format_args!
    t!(format!("{0:x} {0:X}", 15), "f F");
    t!(format!("{0:x} {0:X} {}", 15), "f F 15");
    // NOTE: For now the longer test cases must not be followed immediately by
    // >1 empty lines, or the pretty printer will break. Since no one wants to
    // touch the current pretty printer (#751), we have no choice but to work
    // around it. Some of the following test cases are also affected.
    t!(format!("{:x}{0:X}{a:x}{:X}{1:x}{a:X}", 13, 14, a=15), "dDfEeF");
    t!(format!("{a:x} {a:X}", a=15), "f F");

    // And its edge cases
    t!(format!("{a:.0$} {b:.0$} {0:.0$}\n{a:.c$} {b:.c$} {c:.c$}",
               4, a="abcdefg", b="hijklmn", c=3),
               "abcd hijk 4\nabc hij 3");
    t!(format!("{a:.*} {0} {:.*}", 4, 3, "efgh", a="abcdef"), "abcd 4 efg");
    t!(format!("{:.a$} {a} {a:#x}", "aaaaaa", a=2), "aa 2 0x2");


    // Test that pointers don't get truncated.
    {
        let val = usize::MAX;
        let exp = format!("{:#x}", val);
        t!(format!("{:p}", val as *const isize), exp);
    }

    // Escaping
    t!(format!("{{"), "{");
    t!(format!("}}"), "}");

    test_write();
    test_print();
    test_order();
    test_once();

    // make sure that format! doesn't move out of local variables
    let a: Box<_> = box 3;
    format!("{}", a);
    format!("{}", a);

    // make sure that format! doesn't cause spurious unused-unsafe warnings when
    // it's inside of an outer unsafe block
    unsafe {
        let a: isize = ::std::mem::transmute(3_usize);
        format!("{}", a);
    }

    test_format_args();

    // test that trailing commas are acceptable
    format!("{}", "test",);
    format!("{foo}", foo="test",);

    test_refcell();
}

// Basic test to make sure that we can invoke the `write!` macro with an
// fmt::Write instance.
fn test_write() {
    let mut buf = String::new();
    write!(&mut buf, "{}", 3);
    {
        let w = &mut buf;
        write!(w, "{foo}", foo=4);
        write!(w, "{}", "hello");
        writeln!(w, "{}", "line");
        writeln!(w, "{foo}", foo="bar");
        w.write_char('☃');
        w.write_str("str");
    }

    t!(buf, "34helloline\nbar\n☃str");
}

// Just make sure that the macros are defined, there's not really a lot that we
// can do with them just yet (to test the output)
fn test_print() {
    print!("hi");
    print!("{:?}", vec![0u8]);
    println!("hello");
    println!("this is a {}", "test");
    println!("{foo}", foo="bar");
}

// Just make sure that the macros are defined, there's not really a lot that we
// can do with them just yet (to test the output)
fn test_format_args() {
    let mut buf = String::new();
    {
        let w = &mut buf;
        write!(w, "{}", format_args!("{}", 1));
        write!(w, "{}", format_args!("test"));
        write!(w, "{}", format_args!("{test}", test=3));
    }
    let s = buf;
    t!(s, "1test3");

    let s = fmt::format(format_args!("hello {}", "world"));
    t!(s, "hello world");
    let s = format!("{}: {}", "args were", format_args!("hello {}", "world"));
    t!(s, "args were: hello world");
}

fn test_order() {
    // Make sure format!() arguments are always evaluated in a left-to-right
    // ordering
    fn foo() -> isize {
        static mut FOO: isize = 0;
        unsafe {
            FOO += 1;
            FOO
        }
    }
    assert_eq!(format!("{} {} {a} {b} {} {c}",
                       foo(), foo(), foo(), a=foo(), b=foo(), c=foo()),
               "1 2 4 5 3 6".to_string());
}

fn test_once() {
    // Make sure each argument are evaluated only once even though it may be
    // formatted multiple times
    fn foo() -> isize {
        static mut FOO: isize = 0;
        unsafe {
            FOO += 1;
            FOO
        }
    }
    assert_eq!(format!("{0} {0} {0} {a} {a} {a}", foo(), a=foo()),
               "1 1 1 2 2 2".to_string());
}

fn test_refcell() {
    let refcell = RefCell::new(5);
    assert_eq!(format!("{:?}", refcell), "RefCell { value: 5 }");
    let borrow = refcell.borrow_mut();
    assert_eq!(format!("{:?}", refcell), "RefCell { value: <borrowed> }");
    drop(borrow);
    assert_eq!(format!("{:?}", refcell), "RefCell { value: 5 }");
}
