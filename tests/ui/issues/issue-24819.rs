use std::collections::HashSet;

fn main() {
    let mut v = Vec::new();
    foo(&mut v);
    //~^ ERROR mismatched types
    //~| expected `&mut HashSet<u32>`, found `&mut Vec<_>`
}

fn foo(h: &mut HashSet<u32>) {
}
