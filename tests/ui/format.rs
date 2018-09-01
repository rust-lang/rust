#![feature(tool_lints)]
#![allow(clippy::print_literal)]
#![warn(clippy::useless_format)]

struct Foo(pub String);

macro_rules! foo {
  ($($t:tt)*) => (Foo(format!($($t)*)))
}

fn main() {
    format!("foo");

    format!("{}", "foo");
    format!("{:?}", "foo"); // don't warn about debug
    format!("{:8}", "foo");
    format!("{:+}", "foo"); // warn when the format makes no difference
    format!("{:<}", "foo"); // warn when the format makes no difference
    format!("foo {}", "bar");
    format!("{} bar", "foo");

    let arg: String = "".to_owned();
    format!("{}", arg);
    format!("{:?}", arg); // don't warn about debug
    format!("{:8}", arg);
    format!("{:+}", arg); // warn when the format makes no difference
    format!("{:<}", arg); // warn when the format makes no difference
    format!("foo {}", arg);
    format!("{} bar", arg);

    // we don’t want to warn for non-string args, see #697
    format!("{}", 42);
    format!("{:?}", 42);
    format!("{:+}", 42);
    format!("foo {}", 42);
    format!("{} bar", 42);

    // we only want to warn about `format!` itself
    println!("foo");
    println!("{}", "foo");
    println!("foo {}", "foo");
    println!("{}", 42);
    println!("foo {}", 42);

    // A format! inside a macro should not trigger a warning
    foo!("should not warn");
}
