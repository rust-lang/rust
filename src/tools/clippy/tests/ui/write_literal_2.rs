#![allow(unused_must_use)]
#![warn(clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    writeln!(&mut v, "{}", "{hello}");
    writeln!(&mut v, r"{}", r"{hello}");
    writeln!(&mut v, "{}", '\'');
    writeln!(&mut v, "{}", '"');
    writeln!(&mut v, r"{}", '"'); // don't lint
    writeln!(&mut v, r"{}", '\'');
    writeln!(
        &mut v,
        "some {}",
        "hello \
        world!"
    );
    writeln!(
        &mut v,
        "some {}\
        {} \\ {}",
        "1", "2", "3",
    );
}
