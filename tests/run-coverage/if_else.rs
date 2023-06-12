#![allow(unused_assignments, unused_variables)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;

    let mut countdown = 0;
    if
        is_true
    {
        countdown
        =
            10
        ;
    }
    else // Note coverage region difference without semicolon
    {
        countdown
        =
            100
    }

    if
        is_true
    {
        countdown
        =
            10
        ;
    }
    else
    {
        countdown
        =
            100
        ;
    }
}
