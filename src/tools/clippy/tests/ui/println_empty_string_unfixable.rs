#![allow(clippy::match_single_binding)]

// If there is a comment in the span of macro call, we don't provide an auto-fix suggestion.
#[rustfmt::skip]
fn issue_16167() {
    //~v println_empty_string
    println!("" /* comment */);
    //~v println_empty_string
    eprintln!("" /* comment */);

    //~v println_empty_string
    println!( // comment
                "");
    //~v println_empty_string
    eprintln!( // comment
                "");

    //~v println_empty_string
    println!("", /* comment */);

    //~v println_empty_string
    println!(
        "\
            \
            ",

    // there is a comment in the macro span regardless of its position

    );
}
