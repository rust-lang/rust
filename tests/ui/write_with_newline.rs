#![allow(write_literal)]
#![warn(write_with_newline)]

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
}
