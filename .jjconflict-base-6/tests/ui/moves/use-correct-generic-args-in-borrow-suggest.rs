//! Regression test for #145164: For normal calls, make sure the suggestion to borrow generic inputs
//! uses the generic args from the callee's type rather than those attached to the callee's HIR
//! node. In cases where the callee isn't an identifier expression, its HIR node won't have its
//! generic arguments attached, which could lead to ICE when it had other generic args. In this
//! case, the callee expression is `run.clone()`, to which `clone`'s generic arguments are attached.

fn main() {
    let value = String::new();
    run.clone()(value, ());
    run(value, ());
    //~^ ERROR use of moved value: `value`
}
fn run<F, T: Clone>(value: T, f: F) {}
