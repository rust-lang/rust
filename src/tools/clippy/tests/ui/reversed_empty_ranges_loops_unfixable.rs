#![warn(clippy::reversed_empty_ranges)]
#![allow(clippy::uninlined_format_args)]

fn main() {
    for i in 5..5 {
        //~^ reversed_empty_ranges

        println!("{}", i);
    }

    for i in (5 + 2)..(8 - 1) {
        //~^ reversed_empty_ranges

        println!("{}", i);
    }
}
