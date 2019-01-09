#![allow(unused)]

fn foo<F>(f: F)
    where F: Fn()
{
}

fn main() {
    // Test that this closure is inferred to `FnOnce` because it moves
    // from `y.0`. This affects the error output (the error is that
    // the closure implements `FnOnce`, not that it moves from inside
    // a `Fn` closure.)
    let y = (vec![1, 2, 3], 0);
    let c = || drop(y.0); //~ ERROR expected a closure that implements the `Fn` trait
    foo(c);
}
