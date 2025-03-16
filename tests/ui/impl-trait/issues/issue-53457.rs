//@ check-pass

#![feature(type_alias_impl_trait)]
type X = impl Clone;

fn bar<F: Fn(&i32) + Clone>(f: F) -> F {
    f
}

#[define_opaque(X)]
fn foo() -> X {
    bar(|_| ())
}

fn main() {}
