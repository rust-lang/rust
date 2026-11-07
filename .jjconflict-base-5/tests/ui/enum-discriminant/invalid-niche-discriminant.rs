//@ needs-rustc-debug-assertions
//@ revisions: normal with_delayed
//@ [with_delayed] compile-flags: -Z eagerly-emit-delayed-bugs

#![crate_type = "lib"]

// Repro for <https://github.com/rust-lang/rust/issues/144501>
// which ICEd because the calculated layout is invalid
// but which we needn't care about as the discriminant already was.

enum E {
//~^ ERROR must be specified
//[with_delayed]~| ERROR variant 1 has discriminant 3
    S0 {
        s: String,
    },
    Bar = {
        let x = 1;
        3
    },
}

static C: E = E::S1 { u: 23 };
//~^ ERROR no variant named
//[with_delayed]~| ERROR but no error emitted
