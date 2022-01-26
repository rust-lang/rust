#![allow(unused_must_use)]
#![warn(clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    writeln!(v, "{}", "{hello}");
    writeln!(v, r"{}", r"{hello}");
    writeln!(v, "{}", '\'');
    writeln!(v, "{}", '"');
    writeln!(v, r"{}", '"'); // don't lint
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
}
