#![allow(unused_must_use)]
#![warn(clippy::needless_raw_strings, clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    writeln!(v, "{}", "{hello}");
    writeln!(v, r"{}", r"{hello}");
    writeln!(v, "{}", '\'');
    writeln!(v, "{}", '"');
    writeln!(v, r"{}", '"');
    writeln!(v, r"{}", '\'');
    writeln!(
        v,
        "some {}",
        "hello \
        world!"
    );
    writeln!(
        v,
        "some {}\
        {} \\ {}",
        "1", "2", "3",
    );
    writeln!(v, "{}", "\\");
    writeln!(v, r"{}", "\\");
    writeln!(v, r#"{}"#, "\\");
    writeln!(v, "{}", r"\");
    writeln!(v, "{}", "\r");
    writeln!(v, r#"{}{}"#, '#', '"'); // hard mode
    writeln!(v, r"{}", "\r"); // should not lint
}
