#![allow(unused_assignments, unused_variables)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;

    let (mut a, mut b, mut c) = (0, 0, 0);
    if is_true {
        a = 1;
        b = 10;
        c = 100;
    }
    let
        somebool
        =
            a < b
        ||
            b < c
    ;
    let
        somebool
        =
            b < a
        ||
            b < c
    ;
    let somebool = a < b && b < c;
    let somebool = b < a && b < c;

    if
        !
        is_true
    {
        a = 2
        ;
    }

    if
        is_true
    {
        b = 30
        ;
    }
    else
    {
        c = 400
        ;
    }

    if !is_true {
        a = 2;
    }

    if is_true {
        b = 30;
    } else {
        c = 400;
    }
}
