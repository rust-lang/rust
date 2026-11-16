// Issue #84346.
use std::fmt::Display;

fn main() {
    let x = vec![1, 2, 3];
    let y: Vec<dyn Display> = x.into_iter().collect();
    //~^ ERROR: the size for values of type `dyn std::fmt::Display` cannot be known at compilation time
    //~| ERROR: a value of type `Vec<dyn std::fmt::Display>` cannot be built from an iterator over elements of type `{integer}`
}
