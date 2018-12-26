// aux-build:augmented_assignments.rs

extern crate augmented_assignments;

use augmented_assignments::Int;

fn main() {
    let mut x = Int(0);
    x += 1;
}
