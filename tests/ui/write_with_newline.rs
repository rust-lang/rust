// FIXME: Ideally these suggestions would be fixed via rustfix. Blocked by rust-lang/rust#53934
// // run-rustfix

#![allow(clippy::write_literal)]
#![warn(clippy::write_with_newline)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    write!(&mut v, "Hello\n");
    write!(&mut v, "Hello {}\n", "world");
    write!(&mut v, "Hello {} {}\n", "world", "#2");
    write!(&mut v, "{}\n", 1265);

    // These should be fine
    write!(&mut v, "");
    write!(&mut v, "Hello");
    writeln!(&mut v, "Hello");
    writeln!(&mut v, "Hello\n");
    writeln!(&mut v, "Hello {}\n", "world");
    write!(&mut v, "Issue\n{}", 1265);
    write!(&mut v, "{}", 1265);
    write!(&mut v, "\n{}", 1275);
    write!(&mut v, "\n\n");
    write!(&mut v, "like eof\n\n");
    write!(&mut v, "Hello {} {}\n\n", "world", "#2");
    writeln!(&mut v, "\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    writeln!(&mut v, "\nbla\n\n"); // #3126

    // Escaping
    write!(&mut v, "\\n"); // #3514
    write!(&mut v, "\\\n"); // should fail
    write!(&mut v, "\\\\n");

    // Raw strings
    write!(&mut v, r"\n"); // #3778

    // Literal newlines should also fail
    write!(
        &mut v,
        "
"
    );
    write!(
        &mut v,
        r"
"
    );

    // Don't warn on CRLF (#4208)
    write!(&mut v, "\r\n");
    write!(&mut v, "foo\r\n");
    write!(&mut v, "\\r\n"); //~ ERROR
    write!(&mut v, "foo\rbar\n");
}
