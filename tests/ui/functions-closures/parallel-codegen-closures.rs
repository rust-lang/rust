//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(stable_features)]

// Tests parallel codegen - this can fail if the symbol for the anonymous
// closure in `sum` pollutes the second codegen unit from the first.

//@ compile-flags: -C codegen_units=2

#![feature(iter_arith)]

mod a {
    fn foo() {
        let x = ["a", "bob", "c"];
        let len: usize = x.iter().map(|s| s.len()).sum();
    }
}

mod b {
    fn bar() {
        let x = ["a", "bob", "c"];
        let len: usize = x.iter().map(|s| s.len()).sum();
    }
}

fn main() {
}
