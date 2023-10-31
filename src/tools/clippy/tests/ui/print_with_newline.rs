// FIXME: Ideally these suggestions would be fixed via rustfix. Blocked by rust-lang/rust#53934
//

#![allow(clippy::print_literal)]
#![warn(clippy::print_with_newline)]

fn main() {
    print!("Hello\n");
    print!("Hello {}\n", "world");
    print!("Hello {} {}\n", "world", "#2");
    print!("{}\n", 1265);
    print!("\n");

    // these are all fine
    print!("");
    print!("Hello");
    println!("Hello");
    println!("Hello\n");
    println!("Hello {}\n", "world");
    print!("Issue\n{}", 1265);
    print!("{}", 1265);
    print!("\n{}", 1275);
    print!("\n\n");
    print!("like eof\n\n");
    print!("Hello {} {}\n\n", "world", "#2");
    println!("\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    println!("\nbla\n\n"); // #3126

    // Escaping
    print!("\\n"); // #3514
    print!("\\\n"); // should fail
    print!("\\\\n");

    // Raw strings
    print!(r"\n"); // #3778

    // Literal newlines should also fail
    print!(
        "
"
    );
    print!(
        r"
"
    );

    // Don't warn on CRLF (#4208)
    print!("\r\n");
    print!("foo\r\n");
    print!("\\r\n"); // should fail
    print!("foo\rbar\n");

    // Ignore expanded format strings
    macro_rules! newline {
        () => {
            "\n"
        };
    }
    print!(newline!());
}
