// FIXME: Ideally these suggestions would be fixed via rustfix. Blocked by rust-lang/rust#53934
// // run-rustfix

#![allow(clippy::write_literal)]
#![warn(clippy::write_with_newline)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    write!(v, "Hello\n");
    write!(v, "Hello {}\n", "world");
    write!(v, "Hello {} {}\n", "world", "#2");
    write!(v, "{}\n", 1265);
    write!(v, "\n");

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
    writeln!(v, "\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    writeln!(v, "\nbla\n\n"); // #3126

    // Escaping
    write!(v, "\\n"); // #3514
    write!(v, "\\\n"); // should fail
    write!(v, "\\\\n");

    // Raw strings
    write!(v, r"\n"); // #3778

    // Literal newlines should also fail
    write!(
        v,
        "
"
    );
    write!(
        v,
        r"
"
    );

    // Don't warn on CRLF (#4208)
    write!(v, "\r\n");
    write!(v, "foo\r\n");
    write!(v, "\\r\n"); //~ ERROR
    write!(v, "foo\rbar\n");
}
