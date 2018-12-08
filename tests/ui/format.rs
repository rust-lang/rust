// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


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
    format!("{:width$}", "foo", width = 8);
    format!("{:+}", "foo"); // warn when the format makes no difference
    format!("{:<}", "foo"); // warn when the format makes no difference
    format!("foo {}", "bar");
    format!("{} bar", "foo");

    let arg: String = "".to_owned();
    format!("{}", arg);
    format!("{:?}", arg); // don't warn about debug
    format!("{:8}", arg);
    format!("{:width$}", arg, width = 8);
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

    // precision on string means slicing without panicking on size:
    format!("{:.1}", "foo"); // could be "foo"[..1]
    format!("{:.10}", "foo"); // could not be "foo"[..10]
    format!("{:.prec$}", "foo", prec = 1);
    format!("{:.prec$}", "foo", prec = 10);

    format!("{}", 42.to_string());
    let x = std::path::PathBuf::from("/bar/foo/qux");
    format!("{}", x.display().to_string());
}
