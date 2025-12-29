#![allow(unused_must_use)]
#![warn(clippy::writeln_empty_string)]

use std::io::Write;

// If there is a comment in the span of macro call, we don't provide an auto-fix suggestion.
#[rustfmt::skip]
fn issue_16251() {
    let mut v = Vec::new();

    writeln!(v, /* comment */ "");
    //~^ writeln_empty_string

    writeln!(v, "" /* comment */);
    //~^ writeln_empty_string

    //~v writeln_empty_string
    writeln!(v,
        "\
            \
            "

    // there is a comment in the macro span regardless of its position

    );
}
