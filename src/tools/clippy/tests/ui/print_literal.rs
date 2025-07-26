#![warn(clippy::print_literal)]
#![allow(clippy::uninlined_format_args, clippy::literal_string_with_formatting_args)]

fn main() {
    // these should be fine
    print!("Hello");
    println!("Hello");
    let world = "world";
    println!("Hello {}", world);
    println!("Hello {world}", world = world);
    println!("3 in hex is {:X}", 3);
    println!("2 + 1 = {:.4}", 3);
    println!("2 + 1 = {:5.4}", 3);
    println!("Debug test {:?}", "hello, world");
    println!("{0:8} {1:>8}", "hello", "world");
    println!("{1:8} {0:>8}", "hello", "world");
    println!("{foo:8} {bar:>8}", foo = "hello", bar = "world");
    println!("{bar:8} {foo:>8}", foo = "hello", bar = "world");
    println!("{number:>width$}", number = 1, width = 6);
    println!("{number:>0width$}", number = 1, width = 6);
    println!("{} of {:b} people know binary, the other half doesn't", 1, 2);
    println!("10 / 4 is {}", 2.5);
    println!("2 + 1 = {}", 3);
    println!("From expansion {}", stringify!(not a string literal));

    // these should throw warnings
    print!("Hello {}", "world");
    //~^ print_literal

    println!("Hello {} {}", world, "world");
    //~^ print_literal

    println!("Hello {}", "world");
    //~^ print_literal

    println!("{} {:.4}", "a literal", 5);
    //~^ print_literal

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    println!("{0} {1}", "hello", "world");
    //~^ print_literal

    println!("{1} {0}", "hello", "world");
    //~^ print_literal

    // named args shouldn't change anything either
    println!("{foo} {bar}", foo = "hello", bar = "world");
    //~^ print_literal

    println!("{bar} {foo}", foo = "hello", bar = "world");
    //~^ print_literal

    // The string literal from `file!()` has a callsite span that isn't marked as coming from an
    // expansion
    println!("file: {}", file!());

    // Braces in unicode escapes should not be escaped
    println!("{}", "{} \x00 \u{ab123} \\\u{ab123} {:?}");
    //~^ print_literal
    println!("{}", "\\\u{1234}");
    //~^ print_literal
    // This does not lint because it would have to suggest unescaping the character
    println!(r"{}", "\u{ab123}");
    // These are not unicode escapes
    println!("{}", r"\u{ab123} \u{{");
    //~^ print_literal
    println!(r"{}", r"\u{ab123} \u{{");
    //~^ print_literal
    println!("{}", r"\{ab123} \u{{");
    //~^ print_literal
    println!("{}", "\\u{ab123}");
    //~^ print_literal
    println!("{}", "\\\\u{1234}");
    //~^ print_literal

    println!("mixed: {} {world}", "{hello}");
    //~^ print_literal
}

fn issue_13959() {
    println!("{}", r#"""#);
    //~^ print_literal
    println!(
        "{}",
        r#"
        //~^ print_literal
        foo
        \
        \\
        "
        \"
        bar
"#
    );
}

fn issue_14930() {
    println!("Hello {3} is {0:2$.1$}", 0.01, 2, 3, "x");
    //~^ print_literal
    println!("Hello {2} is {0:3$.1$}", 0.01, 2, "x", 3);
    //~^ print_literal
    println!("Hello {1} is {0:3$.2$}", 0.01, "x", 2, 3);
    //~^ print_literal
    println!("Hello {0} is {1:3$.2$}", "x", 0.01, 2, 3);
    //~^ print_literal
}
