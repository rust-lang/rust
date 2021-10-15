// run-rustfix

#![allow(clippy::print_literal, clippy::redundant_clone, clippy::to_string_in_format_args)]
#![warn(clippy::useless_format)]

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

    format!("{}", "foo");
    format!("{:?}", "foo"); // Don't warn about `Debug`.
    format!("{:8}", "foo");
    format!("{:width$}", "foo", width = 8);
    format!("{:+}", "foo"); // Warn when the format makes no difference.
    format!("{:<}", "foo"); // Warn when the format makes no difference.
    format!("foo {}", "bar");
    format!("{} bar", "foo");

    let arg: String = "".to_owned();
    format!("{}", arg);
    format!("{:?}", arg); // Don't warn about debug.
    format!("{:8}", arg);
    format!("{:width$}", arg, width = 8);
    format!("{:+}", arg); // Warn when the format makes no difference.
    format!("{:<}", arg); // Warn when the format makes no difference.
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
}
