// run-pass

#![feature(existential_type)]

existential type X: Clone;

fn bar<F: Fn(&i32) + Clone>(f: F) -> F {
    f
}

fn foo() -> X {
    bar(|x| ())
}

fn main() {}
