//! Regression test for https://github.com/rust-lang/rust/issues/153205.
//! An uninitialized recursive GAT projection used to hang during optimized MIR processing
//! instead of reporting the layout cycle.

//@ build-fail
//@ compile-flags: -O

trait Apply {
    type Output<T>;
}

struct Identity;

impl Apply for Identity {
    type Output<T> = T;
}

struct Thing<A: Apply>(A::Output<Self>);
//~^ ERROR cycle detected when computing layout of `Thing<Identity>`

fn foo<A: Apply>() {
    let _x: Thing<A>;
}

fn main() {
    foo::<Identity>();
}
