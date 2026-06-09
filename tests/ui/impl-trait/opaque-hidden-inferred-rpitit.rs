//@ check-pass

// Make sure that the `opaque_hidden_inferred_bound` lint doesn't fire on
// RPITITs with no hidden type.

trait T0 {}

trait T1 {
    type A: Send;
}

trait T2 {
    fn foo() -> impl T1<A = ((), impl T0)>;
}

fn main() {}
