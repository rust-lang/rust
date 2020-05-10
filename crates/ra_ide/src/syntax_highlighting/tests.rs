use std::fs;

use test_utils::{assert_eq_text, project_dir, read_text};

use crate::{
    mock_analysis::{single_file, MockAnalysis},
    FileRange, TextRange,
};

#[test]
fn test_highlighting() {
    let (analysis, file_id) = single_file(
        r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
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

    let mut x = 42;
    let y = &mut x;
    let z = &y;

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
    );
    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/highlighting.html");
    let actual_html = &analysis.highlight_as_html(file_id, false).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
}

#[test]
fn test_rainbow_highlighting() {
    let (analysis, file_id) = single_file(
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
    );
    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/rainbow_highlighting.html");
    let actual_html = &analysis.highlight_as_html(file_id, true).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
}

#[test]
fn accidentally_quadratic() {
    let file = project_dir().join("crates/ra_syntax/test_data/accidentally_quadratic");
    let src = fs::read_to_string(file).unwrap();

    let mut mock = MockAnalysis::new();
    let file_id = mock.add_file("/main.rs", &src);
    let host = mock.analysis_host();

    // let t = std::time::Instant::now();
    let _ = host.analysis().highlight(file_id).unwrap();
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
            }"#,
    );

    // The "x"
    let highlights = &analysis
        .highlight_range(FileRange { file_id, range: TextRange::at(82.into(), 1.into()) })
        .unwrap();

    assert_eq!(&highlights[0].highlight.to_string(), "field.declaration");
}

#[test]
fn test_flattening() {
    let (analysis, file_id) = single_file(
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
    );

    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/highlight_injection.html");
    let actual_html = &analysis.highlight_as_html(file_id, false).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
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
    let (analysis, file_id) = single_file(
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

    println!("{\x41}", A = 92);
    println!("{ничоси}", ничоси = 92);
}"#
        .trim(),
    );

    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/highlight_strings.html");
    let actual_html = &analysis.highlight_as_html(file_id, false).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
}
