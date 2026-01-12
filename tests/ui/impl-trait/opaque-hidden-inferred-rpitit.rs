// Make sure that the `opaque_hidden_inferred_bound` lint doesn't fire on
// RPITITs with no hidden type.
// This may be no longer necessary since we check the bounds of projection
// in definitions eagerly now.

trait T0 {}

trait T1 {
    type A: Send;
}

trait T2 {
    fn foo() -> impl T1<A = ((), impl T0)>;
    //~^ ERROR: `impl T0` cannot be sent between threads safely [E0277]
}

fn main() {}
