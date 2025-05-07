//@no-rustfix: overlapping suggestions
#![allow(unused_must_use)]
#![warn(clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    writeln!(v, "{}", "{hello}");
    //~^ write_literal

    writeln!(v, r"{}", r"{hello}");
    //~^ write_literal

    writeln!(v, "{}", '\'');
    //~^ write_literal

    writeln!(v, "{}", '"');
    //~^ write_literal

    writeln!(v, r"{}", '"');
    //~^ write_literal

    writeln!(v, r"{}", '\'');
    //~^ write_literal

    writeln!(
        v,
        "some {}",
        "hello \
        //~^ write_literal
        world!",
    );
    writeln!(
        v,
        "some {}\
        {} \\ {}",
        "1",
        "2",
        "3",
        //~^^^ write_literal
    );
    writeln!(v, "{}", "\\");
    //~^ write_literal

    writeln!(v, r"{}", "\\");
    //~^ write_literal

    writeln!(v, r#"{}"#, "\\");
    //~^ write_literal

    writeln!(v, "{}", r"\");
    //~^ write_literal

    writeln!(v, "{}", "\r");
    //~^ write_literal

    // hard mode
    writeln!(v, r#"{}{}"#, '#', '"');
    //~^ write_literal

    // should not lint
    writeln!(v, r"{}", "\r");
}
