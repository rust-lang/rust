//@ check-pass
//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates

// Regression test for #153205.
// Tests that SROA does not loop forever on self-referential types.

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
