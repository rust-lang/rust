// Shouldn't highlight `let x = 1` as having type bool.
//@ edition:2024

fn main() {
    if let x = 1 && 2 {}
    //~^ ERROR mismatched types
}
