//@ check-pass

#![feature(precise_capturing_of_types)]
//~^ WARN the feature `precise_capturing_of_types` is incomplete

fn uncaptured<T>() -> impl Sized + use<> {}

fn outlives_static<T: 'static>(_: T) {}

fn foo<'a>() {
    outlives_static(uncaptured::<&'a ()>());
}

fn main() {}
