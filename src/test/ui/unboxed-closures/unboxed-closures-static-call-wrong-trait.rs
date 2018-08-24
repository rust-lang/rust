#![feature(unboxed_closures)]

fn to_fn_mut<A,F:FnMut<A>>(f: F) -> F { f }

fn main() {
    let mut_ = to_fn_mut(|x| x);
    mut_.call((0, )); //~ ERROR no method named `call` found
}
