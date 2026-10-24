// This test serves to document the change in semantics introduced by
// rust-lang/rust#138961, extended to `if let` closure captures.
//
// A corollary of partial-pattern.rs: while the tuple access testcase makes
// it clear why these semantics are useful, it is actually the dereference
// being performed by the pattern that matters.
//
// Before rust-lang/rust#154210, `if let` in closures captured all of `x`, so
// this test did not fail because the closure is never called.
//@normalize-stderr-test: "constructing invalid value of type [^:]+:" -> "constructing invalid value:"

#![allow(irrefutable_let_patterns)]

fn main() {
    // the inner reference is dangling
    let x: &&u32 = unsafe {
        let x: u32 = 42;
        &&*&raw const x
    };

    //~v ERROR: encountered a dangling reference
    let _ = || {
        if let &&_y = x {}
    };
}
