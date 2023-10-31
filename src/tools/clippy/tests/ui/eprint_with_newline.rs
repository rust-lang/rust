#![allow(clippy::print_literal)]
#![warn(clippy::print_with_newline)]

fn main() {
    eprint!("Hello\n");
    eprint!("Hello {}\n", "world");
    eprint!("Hello {} {}\n", "world", "#2");
    eprint!("{}\n", 1265);
    eprint!("\n");

    // these are all fine
    eprint!("");
    eprint!("Hello");
    eprintln!("Hello");
    eprintln!("Hello\n");
    eprintln!("Hello {}\n", "world");
    eprint!("Issue\n{}", 1265);
    eprint!("{}", 1265);
    eprint!("\n{}", 1275);
    eprint!("\n\n");
    eprint!("like eof\n\n");
    eprint!("Hello {} {}\n\n", "world", "#2");
    eprintln!("\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    eprintln!("\nbla\n\n"); // #3126

    // Escaping
    eprint!("\\n"); // #3514
    eprint!("\\\n"); // should fail
    eprint!("\\\\n");

    // Raw strings
    eprint!(r"\n"); // #3778

    // Literal newlines should also fail
    eprint!(
        "
"
    );
    eprint!(
        r"
"
    );

    // Don't warn on CRLF (#4208)
    eprint!("\r\n");
    eprint!("foo\r\n");
    eprint!("\\r\n");
    eprint!("foo\rbar\n");

    // Ignore expanded format strings
    macro_rules! newline {
        () => {
            "\n"
        };
    }
    eprint!(newline!());
}
