// Tests parallel codegen - this can fail if the symbol for the anonymous
// closure in `sum` pollutes the second codegen unit from the first.

// ignore-bitrig
// compile-flags: -C codegen_units=2

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
