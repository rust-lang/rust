//! Diagnostic test for https://github.com/rust-lang/rust/issues/164504

fn foo<T: Clone>(a: &T) {
    let f = |_a| {};
    //~^ HELP consider annotating the closure parameter as a reference so it is generic over the reference's lifetime
    f(a);
    (|a| f(&a))(a.clone());
    //~^ ERROR `a` does not live long enough
}

fn main() {}
