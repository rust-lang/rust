#![warn(clippy::reversed_empty_ranges)]

fn main() {
    for i in 5..5 {
        println!("{}", i);
    }

    for i in (5 + 2)..(8 - 1) {
        println!("{}", i);
    }
}
