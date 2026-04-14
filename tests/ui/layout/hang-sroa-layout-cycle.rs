//@ build-fail
//~^ ERROR cycle detected when computing layout of `Thing<Identity>`
//@ compile-flags: -Copt-level=3

// Regression test for https://github.com/rust-lang/rust/issues/153205:
// a struct that contains itself via an associated type used to cause the
// `ScalarReplacementOfAggregates` MIR pass to loop forever after the
// layout-cycle error was emitted. The SROA pass now bounds its iteration
// count, so the compile terminates with the expected layout-cycle error
// rather than hanging.

trait Apply {
    type Output<T>;
}

struct Identity;

impl Apply for Identity {
    type Output<T> = T;
}

struct Thing<A: Apply>(A::Output<Self>);

fn foo<A: Apply>() {
    let _x: Thing<A>;
}

fn main() {
    foo::<Identity>();
}
