// Regression test for #54593: the MIR type checker was going wrong
// when a closure returns the `impl Copy` from its parent fn. It was
// (incorrectly) replacing the `impl Copy` in its return type with the
// hidden type (`()`) but that type resulted from a recursive call to
// `foo` and hence is treated opaquely within the closure body.  This
// resulted in a failed subtype relationship.
//
//@ check-pass

fn foo() -> impl Copy { || foo(); }
fn bar() -> impl Copy { || bar(); }
fn main() { }
