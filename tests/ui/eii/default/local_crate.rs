//@ run-pass
//@ check-run-results
#![feature(eii)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    println!("default {x}");
}

fn main() {
    decl1(4);
}
