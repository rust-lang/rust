use std::collections::HashSet;

fn main() {
    let mut v = Vec::new();
    foo(&mut v);
    //~^ ERROR arguments to this function are incorrect
    //~| expected struct `HashSet`, found struct `Vec`
}

fn foo(h: &mut HashSet<u32>) {
}
