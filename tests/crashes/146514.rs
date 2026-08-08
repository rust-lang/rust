//@ known-bug: rust-lang/rust#146514
//@ edition: 2021
//@ compile-flags: --crate-type=lib -Copt-level=0

#![feature(impl_trait_in_assoc_type)]

use core::marker::PhantomData;

struct Emp<T, F> {
    phantom: PhantomData<(*const T, F)>,
}

impl<T, F> Emp<T, F> {
    fn from_fn(_: F) -> Emp<T, F> {
        loop {}
    }

    fn unsize(self) -> Emp<Slice, impl Sized> {
        Emp::from_fn(|| ())
    }
}

trait IntoEmplacable {
    type Closure;

    fn into_emplacable(self) -> Emp<Slice, Self::Closure>;
}

impl<F> IntoEmplacable for Emp<Arr, F> {
    type Closure = impl Sized;

    fn into_emplacable(self) -> Emp<Slice, Self::Closure> {
        self.unsize()
    }
}

impl<F> Into<Emp<Slice, <Emp<Arr, F> as IntoEmplacable>::Closure>> for Emp<Arr, F> {
    fn into(self) -> Emp<Slice, <Emp<Arr, F> as IntoEmplacable>::Closure> {
        self.into_emplacable()
    }
}

fn box_new_with(_: Emp<Slice, impl Sized>) {}

pub struct Arr;
pub struct Slice;

pub fn foo() {
    let e: Emp<Arr, ()> = Emp {
        phantom: PhantomData,
    };
    box_new_with(e.into());
}
