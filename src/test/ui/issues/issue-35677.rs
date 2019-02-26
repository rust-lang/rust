use std::collections::HashSet;

fn is_subset<T>(this: &HashSet<T>, other: &HashSet<T>) -> bool {
    this.is_subset(other)
    //~^ ERROR no method named
}

fn main() {}
