#![allow(unused_assignments, unused_variables)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;
    let is_false = ! is_true;

    let mut some_string = Some(String::from("the string content"));
    println!(
        "The string or alt: {}"
        ,
        some_string
            .
            unwrap_or_else
        (
            ||
            {
                let mut countdown = 0;
                if is_false {
                    countdown = 10;
                }
                "alt string 1".to_owned()
            }
        )
    );

    some_string = Some(String::from("the string content"));
    let
        a
    =
        ||
    {
        let mut countdown = 0;
        if is_false {
            countdown = 10;
        }
        "alt string 2".to_owned()
    };
    println!(
        "The string or alt: {}"
        ,
        some_string
            .
            unwrap_or_else
        (
            a
        )
    );

    some_string = None;
    println!(
        "The string or alt: {}"
        ,
        some_string
            .
            unwrap_or_else
        (
            ||
            {
                let mut countdown = 0;
                if is_false {
                    countdown = 10;
                }
                "alt string 3".to_owned()
            }
        )
    );

    some_string = None;
    let
        a
    =
        ||
    {
        let mut countdown = 0;
        if is_false {
            countdown = 10;
        }
        "alt string 4".to_owned()
    };
    println!(
        "The string or alt: {}"
        ,
        some_string
            .
            unwrap_or_else
        (
            a
        )
    );

    let
        quote_closure
    =
        |val|
    {
        let mut countdown = 0;
        if is_false {
            countdown = 10;
        }
        format!("'{}'", val)
    };
    println!(
        "Repeated, quoted string: {:?}"
        ,
        std::iter::repeat("repeat me")
            .take(5)
            .map
        (
            quote_closure
        )
            .collect::<Vec<_>>()
    );

    let
        _unused_closure
    =
        |
            mut countdown
        |
    {
        if is_false {
            countdown = 10;
        }
        "closure should be unused".to_owned()
    };

    let mut countdown = 10;
    let _short_unused_closure = | _unused_arg: u8 | countdown += 1;

    // Macros can sometimes confuse the coverage results. Compare this next assignment, with an
    // unused closure that invokes the `println!()` macro, with the closure assignment above, that
    // does not use a macro. The closure above correctly shows `0` executions.
    let _short_unused_closure = | _unused_arg: u8 | println!("not called");
    // The closure assignment above is executed, with a line count of `1`, but the `println!()`
    // could not have been called, and yet, there is no indication that it wasn't...

    // ...but adding block braces gives the expected result, showing the block was not executed.
    let _short_unused_closure_block = | _unused_arg: u8 | { println!("not called") };

    let _shortish_unused_closure = | _unused_arg: u8 | {
        println!("not called")
    };

    let _as_short_unused_closure = |
        _unused_arg: u8
    | { println!("not called") };

    let _almost_as_short_unused_closure = |
        _unused_arg: u8
    | { println!("not called") }
    ;
}
