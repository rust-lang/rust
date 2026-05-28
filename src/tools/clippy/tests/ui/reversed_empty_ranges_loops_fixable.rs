#![warn(clippy::reversed_empty_ranges)]
#![allow(clippy::uninlined_format_args)]

fn main() {
    const MAX_LEN: usize = 42;

    for i in 10..0 {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    for i in 10..=0 {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    for i in MAX_LEN..0 {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    for i in 5..=5 {
        // not an error, this is the range with only one element “5”
        println!("{}", i);
    }

    for i in 0..10 {
        // not an error, the start index is less than the end index
        println!("{}", i);
    }

    for i in -10..0 {
        // not an error
        println!("{}", i);
    }

    for i in (10..0).map(|x| x * 2) {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    // testing that the empty range lint folds constants
    for i in 10..5 + 4 {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    for i in (5 + 2)..(3 - 1) {
        //~^ reversed_empty_ranges
        println!("{}", i);
    }

    for i in (2 * 2)..(2 * 3) {
        // no error, 4..6 is fine
        println!("{}", i);
    }

    let x = 42;
    for i in x..10 {
        // no error, not constant-foldable
        println!("{}", i);
    }
}
