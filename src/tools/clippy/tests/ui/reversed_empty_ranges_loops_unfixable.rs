#![warn(clippy::reversed_empty_ranges)]
#![allow(clippy::uninlined_format_args)]

fn main() {
    for i in 5..5 {
        //~^ ERROR: this range is empty so it will yield no values
        //~| NOTE: `-D clippy::reversed-empty-ranges` implied by `-D warnings`
        println!("{}", i);
    }

    for i in (5 + 2)..(8 - 1) {
        //~^ ERROR: this range is empty so it will yield no values
        println!("{}", i);
    }
}
