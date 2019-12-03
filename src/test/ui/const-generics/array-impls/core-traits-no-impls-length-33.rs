pub fn no_debug() {
    println!("{:?}", [0_usize; 33]);
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
    set.insert([0_usize; 33]);
    //~^ ERROR arrays only have std trait implementations for lengths 0..=32
}

pub fn no_partial_eq() -> bool {
    [0_usize; 33] == [1_usize; 33]
    //~^ ERROR binary operation `==` cannot be applied to type `[usize; 33]`
}

pub fn no_partial_ord() -> bool {
    [0_usize; 33] < [1_usize; 33]
    //~^ ERROR binary operation `<` cannot be applied to type `[usize; 33]`
}

pub fn no_into_iterator() {
    for _ in &[0_usize; 33] {
        //~^ ERROR the trait bound `&[usize; 33]: std::iter::IntoIterator` is not satisfied
    }
}

fn main() {}
