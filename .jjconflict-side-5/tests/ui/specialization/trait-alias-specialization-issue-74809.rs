//@ check-pass

#![feature(specialization)]
#![feature(trait_alias)]
#![allow(incomplete_features)]

// Tests that we can specialize on a trait alias.
// Regression test for #74809.

pub trait Marker1<T> {}
pub trait Marker2 {}

pub trait CombinedMarker<T> = Marker1<T> + Marker2;

pub struct Container<T, U> {
    p: std::marker::PhantomData<(T, U)>,
}

pub struct Struct;
impl<T> Marker1<T> for Struct {}

pub trait Trait {
    fn do_thing(&self);
}

impl<T, U: Marker1<T>> Trait for Container<T, U> {
    default fn do_thing(&self) {
        println!("default behavior");
    }
}

impl<T, U: CombinedMarker<T>> Trait for Container<T, U> {
    default fn do_thing(&self) {
        println!("partially specialized behavior");
    }
}

impl<T> Trait for Container<T, Struct> {
    fn do_thing(&self) {
        println!("fully specialized behavior")
    }
}

fn main() {}
