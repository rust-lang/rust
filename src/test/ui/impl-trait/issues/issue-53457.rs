// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type X = impl Clone;

fn bar<F: Fn(&i32) + Clone>(f: F) -> F {
    f
}

fn foo() -> X {
    bar(|_| ())
}

fn main() {}
