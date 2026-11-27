#![feature(unboxed_closures, tuple_trait)]

// Tests that we can't move out of an unboxed closure environment
// if the upvar is captured by ref or the closure takes self by
// reference.

fn to_fn<A:std::marker::Tuple,F:Fn<A>>(f: F) -> F { f }
fn to_fn_mut<A:std::marker::Tuple,F:FnMut<A>>(f: F) -> F { f }
fn to_fn_once<A:std::marker::Tuple,F:FnOnce<A>>(f: F) -> F { f }

fn main() {
    // By-ref cases
    {
        let x = Box::new(0);
        let f = to_fn(|| drop(x)); //~ ERROR cannot move
    }
    {
        let x = Box::new(0);
        let f = to_fn_mut(|| drop(x)); //~ ERROR cannot move
    }
    {
        let x = Box::new(0);
        let f = to_fn_once(|| drop(x)); // OK -- FnOnce
    }
    // By-value cases
    {
        let x = Box::new(0);
        let f = to_fn(move || drop(x)); //~ ERROR cannot move
    }
    {
        let x = Box::new(0);
        let f = to_fn_mut(move || drop(x)); //~ ERROR cannot move
    }
    {
        let x = Box::new(0);
        let f = to_fn_once(move || drop(x)); // this one is ok
    }
}
