#![allow(clippy::print_literal)]
#![warn(clippy::print_with_newline)]

fn main() {
    eprint!("Hello\n");
    //~^ print_with_newline

    eprint!("Hello {}\n", "world");
    //~^ print_with_newline

    eprint!("Hello {} {}\n", "world", "#2");
    //~^ print_with_newline

    eprint!("{}\n", 1265);
    //~^ print_with_newline

    eprint!("\n");
    //~^ print_with_newline

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
    // #3126
    eprintln!("\ndon't\nwarn\nfor\nmultiple\nnewlines\n");
    // #3126
    eprintln!("\nbla\n\n");

    // Escaping
    // #3514
    eprint!("\\n");
    eprint!("\\\n");
    //~^ print_with_newline

    eprint!("\\\\n");

    // Raw strings
    // #3778
    eprint!(r"\n");

    // Literal newlines should also fail
    eprint!(
        //~^ print_with_newline
        "
"
    );
    eprint!(
        //~^ print_with_newline
        r"
"
    );

    // Don't warn on CRLF (#4208)
    eprint!("\r\n");
    eprint!("foo\r\n");
    eprint!("\\r\n");
    //~^ print_with_newline

    eprint!("foo\rbar\n");

    // Ignore expanded format strings
    macro_rules! newline {
        () => {
            "\n"
        };
    }
    eprint!(newline!());
}
