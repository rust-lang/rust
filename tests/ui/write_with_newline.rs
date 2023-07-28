// FIXME: Ideally these suggestions would be fixed via rustfix. Blocked by rust-lang/rust#53934

#![allow(clippy::write_literal)]
#![warn(clippy::write_with_newline)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    write!(v, "Hello\n");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    //~| NOTE: `-D clippy::write-with-newline` implied by `-D warnings`
    write!(v, "Hello {}\n", "world");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    write!(v, "Hello {} {}\n", "world", "#2");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    write!(v, "{}\n", 1265);
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    write!(v, "\n");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline

    // These should be fine
    write!(v, "");
    write!(v, "Hello");
    writeln!(v, "Hello");
    writeln!(v, "Hello\n");
    writeln!(v, "Hello {}\n", "world");
    write!(v, "Issue\n{}", 1265);
    write!(v, "{}", 1265);
    write!(v, "\n{}", 1275);
    write!(v, "\n\n");
    write!(v, "like eof\n\n");
    write!(v, "Hello {} {}\n\n", "world", "#2");
    // #3126
    writeln!(v, "\ndon't\nwarn\nfor\nmultiple\nnewlines\n");
    // #3126
    writeln!(v, "\nbla\n\n");

    // Escaping
    // #3514
    write!(v, "\\n");
    write!(v, "\\\n");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    write!(v, "\\\\n");

    // Raw strings
    // #3778
    write!(v, r"\n");

    // Literal newlines should also fail
    write!(
        //~^ ERROR: using `write!()` with a format string that ends in a single newline
        v,
        "
"
    );
    write!(
        //~^ ERROR: using `write!()` with a format string that ends in a single newline
        v,
        r"
"
    );

    // Don't warn on CRLF (#4208)
    write!(v, "\r\n");
    write!(v, "foo\r\n");
    write!(v, "\\r\n");
    //~^ ERROR: using `write!()` with a format string that ends in a single newline
    write!(v, "foo\rbar\n");

    // Ignore expanded format strings
    macro_rules! newline {
        () => {
            "\n"
        };
    }
    write!(v, newline!());
}
