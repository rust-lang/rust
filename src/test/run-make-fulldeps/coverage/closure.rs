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
}
