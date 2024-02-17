// Checks that an unreachable code warning is emitted when an expression is
// preceded by an expression with an uninhabited type. Previously, the
// variable liveness analysis was "smarter" than the reachability analysis
// in this regard, which led to confusing "unused variable" warnings
// without an accompanying explanatory "unreachable expression" warning.

//@ check-pass

#![warn(unused_variables,unreachable_code)]

enum Foo {}
fn f() -> Foo {todo!()}

fn main() {
    let x = f();
    //~^ WARNING: unused variable: `x`
    let _ = x;
    //~^ WARNING: unreachable expression
}
