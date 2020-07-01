use std::fs;

use expect::{expect_file, ExpectFile};
use test_utils::project_dir;

use crate::{mock_analysis::single_file, FileRange, TextRange};

#[test]
fn test_highlighting() {
    check_highlighting(
        r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
}

trait Bar {
    fn bar(&self) -> i32;
}

impl Bar for Foo {
    fn bar(&self) -> i32 {
        self.x
    }
}

static mut STATIC_MUT: i32 = 0;

fn foo<'a, T>() -> T {
    foo::<'a, i32>()
}

macro_rules! def_fn {
    ($($tt:tt)*) => {$($tt)*}
}

def_fn! {
    fn bar() -> u32 {
        100
    }
}

macro_rules! noop {
    ($expr:expr) => {
        $expr
    }
}

// comment
fn main() {
    println!("Hello, {}!", 92);

    let mut vec = Vec::new();
    if true {
        let x = 92;
        vec.push(Foo { x, y: 1 });
    }
    unsafe {
        vec.set_len(0);
        STATIC_MUT = 1;
    }

    for e in vec {
        // Do nothing
    }

    noop!(noop!(1));

    let mut x = 42;
    let y = &mut x;
    let z = &y;

    let Foo { x: z, y } = Foo { x: z, y };

    y;
}

enum Option<T> {
    Some(T),
    None,
}
use Option::*;

impl<T> Option<T> {
    fn and<U>(self, other: Option<U>) -> Option<(T, U)> {
        match other {
            None => unimplemented!(),
            Nope => Nope,
        }
    }
}
"#
        .trim(),
        expect_file!["crates/ra_ide/test_data/highlighting.html"],
        false,
    );
}

#[test]
fn test_rainbow_highlighting() {
    check_highlighting(
        r#"
fn main() {
    let hello = "hello";
    let x = hello.to_string();
    let y = hello.to_string();

    let x = "other color please!";
    let y = x.to_string();
}

fn bar() {
    let mut hello = "hello";
}
"#
        .trim(),
        expect_file!["crates/ra_ide/test_data/rainbow_highlighting.html"],
        true,
    );
}

#[test]
fn accidentally_quadratic() {
    let file = project_dir().join("crates/ra_syntax/test_data/accidentally_quadratic");
    let src = fs::read_to_string(file).unwrap();

    let (analysis, file_id) = single_file(&src);

    // let t = std::time::Instant::now();
    let _ = analysis.highlight(file_id).unwrap();
    // eprintln!("elapsed: {:?}", t.elapsed());
}

#[test]
fn test_ranges() {
    let (analysis, file_id) = single_file(
        r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
}
"#,
    );

    // The "x"
    let highlights = &analysis
        .highlight_range(FileRange { file_id, range: TextRange::at(45.into(), 1.into()) })
        .unwrap();

    assert_eq!(&highlights[0].highlight.to_string(), "field.declaration");
}

#[test]
fn test_flattening() {
    check_highlighting(
        r##"
fn fixture(ra_fixture: &str) {}

fn main() {
    fixture(r#"
        trait Foo {
            fn foo() {
                println!("2 + 2 = {}", 4);
            }
        }"#
    );
}"##
        .trim(),
        expect_file!["crates/ra_ide/test_data/highlight_injection.html"],
        false,
    );
}

#[test]
fn ranges_sorted() {
    let (analysis, file_id) = single_file(
        r#"
#[foo(bar = "bar")]
macro_rules! test {}
}"#
        .trim(),
    );
    let _ = analysis.highlight(file_id).unwrap();
}

#[test]
fn test_string_highlighting() {
    // The format string detection is based on macro-expansion,
    // thus, we have to copy the macro definition from `std`
    check_highlighting(
        r#"
macro_rules! println {
    ($($arg:tt)*) => ({
        $crate::io::_print($crate::format_args_nl!($($arg)*));
    })
}
#[rustc_builtin_macro]
macro_rules! format_args_nl {
    ($fmt:expr) => {{ /* compiler built-in */ }};
    ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
}

fn main() {
    // from https://doc.rust-lang.org/std/fmt/index.html
    println!("Hello");                 // => "Hello"
    println!("Hello, {}!", "world");   // => "Hello, world!"
    println!("The number is {}", 1);   // => "The number is 1"
    println!("{:?}", (3, 4));          // => "(3, 4)"
    println!("{value}", value=4);      // => "4"
    println!("{} {}", 1, 2);           // => "1 2"
    println!("{:04}", 42);             // => "0042" with leading zerosV
    println!("{1} {} {0} {}", 1, 2);   // => "2 1 1 2"
    println!("{argument}", argument = "test");   // => "test"
    println!("{name} {}", 1, name = 2);          // => "2 1"
    println!("{a} {c} {b}", a="a", b='b', c=3);  // => "a 3 b"
    println!("{{{}}}", 2);                       // => "{2}"
    println!("Hello {:5}!", "x");
    println!("Hello {:1$}!", "x", 5);
    println!("Hello {1:0$}!", 5, "x");
    println!("Hello {:width$}!", "x", width = 5);
    println!("Hello {:<5}!", "x");
    println!("Hello {:-<5}!", "x");
    println!("Hello {:^5}!", "x");
    println!("Hello {:>5}!", "x");
    println!("Hello {:+}!", 5);
    println!("{:#x}!", 27);
    println!("Hello {:05}!", 5);
    println!("Hello {:05}!", -5);
    println!("{:#010x}!", 27);
    println!("Hello {0} is {1:.5}", "x", 0.01);
    println!("Hello {1} is {2:.0$}", 5, "x", 0.01);
    println!("Hello {0} is {2:.1$}", "x", 5, 0.01);
    println!("Hello {} is {:.*}",    "x", 5, 0.01);
    println!("Hello {} is {2:.*}",   "x", 5, 0.01);
    println!("Hello {} is {number:.prec$}", "x", prec = 5, number = 0.01);
    println!("{}, `{name:.*}` has 3 fractional digits", "Hello", 3, name=1234.56);
    println!("{}, `{name:.*}` has 3 characters", "Hello", 3, name="1234.56");
    println!("{}, `{name:>8.*}` has 3 right-aligned characters", "Hello", 3, name="1234.56");
    println!("Hello {{}}");
    println!("{{ Hello");

    println!(r"Hello, {}!", "world");

    // escape sequences
    println!("Hello\nWorld");
    println!("\u{48}\x65\x6C\x6C\x6F World");

    println!("{\x41}", A = 92);
    println!("{ничоси}", ничоси = 92);
}"#
        .trim(),
        expect_file!["crates/ra_ide/test_data/highlight_strings.html"],
        false,
    );
}

#[test]
fn test_unsafe_highlighting() {
    check_highlighting(
        r#"
unsafe fn unsafe_fn() {}

struct HasUnsafeFn;

impl HasUnsafeFn {
    unsafe fn unsafe_method(&self) {}
}

fn main() {
    let x = &5 as *const usize;
    unsafe {
        unsafe_fn();
        HasUnsafeFn.unsafe_method();
        let y = *(x);
        let z = -x;
    }
}
"#
        .trim(),
        expect_file!["crates/ra_ide/test_data/highlight_unsafe.html"],
        false,
    );
}

#[test]
fn test_highlight_doctest() {
    check_highlighting(
        r#"
/// ```
/// let _ = "early doctests should not go boom";
/// ```
struct Foo {
    bar: bool,
}

impl Foo {
    pub const bar: bool = true;

    /// Constructs a new `Foo`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut foo: Foo = Foo::new();
    /// ```
    pub const fn new() -> Foo {
        Foo { bar: true }
    }

    /// `bar` method on `Foo`.
    ///
    /// # Examples
    ///
    /// ```
    /// use x::y;
    ///
    /// let foo = Foo::new();
    ///
    /// // calls bar on foo
    /// assert!(foo.bar());
    ///
    /// let bar = foo.bar || Foo::bar;
    ///
    /// /* multi-line
    ///        comment */
    ///
    /// let multi_line_string = "Foo
    ///   bar
    ///          ";
    ///
    /// ```
    ///
    /// ```rust,no_run
    /// let foobar = Foo::new().bar();
    /// ```
    ///
    /// ```sh
    /// echo 1
    /// ```
    pub fn foo(&self) -> bool {
        true
    }
}

/// ```
/// noop!(1);
/// ```
macro_rules! noop {
    ($expr:expr) => {
        $expr
    }
}
"#
        .trim(),
        expect_file!["crates/ra_ide/test_data/highlight_doctest.html"],
        false,
    );
}

/// Highlights the code given by the `ra_fixture` argument, renders the
/// result as HTML, and compares it with the HTML file given as `snapshot`.
/// Note that the `snapshot` file is overwritten by the rendered HTML.
fn check_highlighting(ra_fixture: &str, expect: ExpectFile, rainbow: bool) {
    let (analysis, file_id) = single_file(ra_fixture);
    let actual_html = &analysis.highlight_as_html(file_id, rainbow).unwrap();
    expect.assert_eq(actual_html)
}
