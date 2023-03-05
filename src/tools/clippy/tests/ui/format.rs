// run-rustfix
// aux-build: proc_macro_with_span.rs
#![warn(clippy::useless_format)]
#![allow(
    unused_tuple_struct_fields,
    clippy::print_literal,
    clippy::redundant_clone,
    clippy::to_string_in_format_args,
    clippy::needless_borrow,
    clippy::uninlined_format_args
)]

extern crate proc_macro_with_span;

struct Foo(pub String);

macro_rules! foo {
    ($($t:tt)*) => (Foo(format!($($t)*)))
}

fn main() {
    format!("foo");
    format!("{{}}");
    format!("{{}} abc {{}}");
    format!(
        r##"foo {{}}
" bar"##
    );

    let _ = format!("");

    format!("{}", "foo");
    format!("{:?}", "foo"); // Don't warn about `Debug`.
    format!("{:8}", "foo");
    format!("{:width$}", "foo", width = 8);
    format!("foo {}", "bar");
    format!("{} bar", "foo");

    let arg = String::new();
    format!("{}", arg);
    format!("{:?}", arg); // Don't warn about debug.
    format!("{:8}", arg);
    format!("{:width$}", arg, width = 8);
    format!("foo {}", arg);
    format!("{} bar", arg);

    // We don’t want to warn for non-string args; see issue #697.
    format!("{}", 42);
    format!("{:?}", 42);
    format!("{:+}", 42);
    format!("foo {}", 42);
    format!("{} bar", 42);

    // We only want to warn about `format!` itself.
    println!("foo");
    println!("{}", "foo");
    println!("foo {}", "foo");
    println!("{}", 42);
    println!("foo {}", 42);

    // A `format!` inside a macro should not trigger a warning.
    foo!("should not warn");

    // Precision on string means slicing without panicking on size.
    format!("{:.1}", "foo"); // Could be `"foo"[..1]`
    format!("{:.10}", "foo"); // Could not be `"foo"[..10]`
    format!("{:.prec$}", "foo", prec = 1);
    format!("{:.prec$}", "foo", prec = 10);

    format!("{}", 42.to_string());
    let x = std::path::PathBuf::from("/bar/foo/qux");
    format!("{}", x.display().to_string());

    // False positive
    let a = "foo".to_string();
    let _ = Some(format!("{}", a + "bar"));

    // Wrap it with braces
    let v: Vec<String> = vec!["foo".to_string(), "bar".to_string()];
    let _s: String = format!("{}", &*v.join("\n"));

    format!("prepend {:+}", "s");

    // Issue #8290
    let x = "foo";
    let _ = format!("{x}");
    let _ = format!("{x:?}"); // Don't lint on debug
    let _ = format!("{y}", y = x);

    // Issue #9234
    let abc = "abc";
    let _ = format!("{abc}");
    let xx = "xx";
    let _ = format!("{xx}");

    // Issue #10148
    println!(proc_macro_with_span::with_span!(""something ""));
}
