#![feature(rustc_attrs)]

// A (reached depth 0)
//   ...
//      B // depth 1 -- reached depth = 0
//          C // depth 2 -- reached depth = 1 (should be 0)
//              B
//          A // depth 0
//   D (reached depth 1)
//      C (cache -- reached depth = 2)

struct A {
    b: B,
    c: C,
}

struct B {
    c: C,
    a: Option<Box<A>>,
}

struct C {
    b: Option<Box<B>>,
}

#[rustc_evaluate_where_clauses]
fn test<X: Send>() {}

fn main() {
    test::<A>();
}
