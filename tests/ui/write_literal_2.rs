//@no-rustfix: overlapping suggestions
#![allow(unused_must_use)]
#![warn(clippy::needless_raw_strings, clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    writeln!(v, "{}", "{hello}");
    //~^ ERROR: literal with an empty format string
    //~| NOTE: `-D clippy::write-literal` implied by `-D warnings`
    writeln!(v, r"{}", r"{hello}");
    //~^ ERROR: unnecessary raw string literal
    //~| NOTE: `-D clippy::needless-raw-strings` implied by `-D warnings`
    //~| ERROR: literal with an empty format string
    writeln!(v, "{}", '\'');
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{}", '"');
    //~^ ERROR: literal with an empty format string
    writeln!(v, r"{}", '"');
    //~^ ERROR: literal with an empty format string
    writeln!(v, r"{}", '\'');
    //~^ ERROR: literal with an empty format string
    writeln!(
        v,
        "some {}",
        "hello \
        //~^ ERROR: literal with an empty format string
        world!"
    );
    writeln!(
        v,
        "some {}\
        {} \\ {}",
        "1",
        "2",
        "3",
        //~^ ERROR: literal with an empty format string
    );
    writeln!(v, "{}", "\\");
    //~^ ERROR: literal with an empty format string
    writeln!(v, r"{}", "\\");
    //~^ ERROR: literal with an empty format string
    writeln!(v, r#"{}"#, "\\");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{}", r"\");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{}", "\r");
    //~^ ERROR: literal with an empty format string
    // hard mode
    writeln!(v, r#"{}{}"#, '#', '"');
    //~^ ERROR: literal with an empty format string
    //~| ERROR: literal with an empty format string
    // should not lint
    writeln!(v, r"{}", "\r");
}
