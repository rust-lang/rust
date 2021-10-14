#![feature(rustc_attrs)]

// Test for a particular corner case where the evaluation
// cache can get out of date. The problem here is that
// when we cache C, we have observed that it reaches
// to depth 2 (the node for B), but we later realize
// that B itself depends on A (reached depth 0). We
// failed to update the depth for C transitively, which
// resulted in an assertion failure when it was referenced
// from D.
//
// A (reached depth 0)
//   E
//      B // depth 2 -- reached depth = 0
//          C // depth 3 -- reached depth = 2 (should be 0)
//              B
//          A // depth 0
//   D (depth 1)
//      C (cache -- reached depth = 2)

struct A {
    e: E,
    d: C,
}

struct E {
    b: B,
}

struct B {
    a: Option<Box<A>>,
    c: C,
}

struct C {
    b: Option<Box<B>>,
}

#[rustc_evaluate_where_clauses]
fn test<X: ?Sized + Send>() {}

fn main() {
    test::<A>();
    //~^ ERROR evaluate(Binder(TraitPredicate(<A as std::marker::Send>, polarity:Positive), [])) = Ok(EvaluatedToOk)
}
