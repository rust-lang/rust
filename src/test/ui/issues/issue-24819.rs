use std::collections::HashSet;

fn main() {
    let mut v = Vec::new();
    foo(&mut v);
    //~^ ERROR mismatched types
    //~| expected struct `std::collections::HashSet`, found struct `std::vec::Vec`
}

fn foo(h: &mut HashSet<u32>) {
}
