#![allow(
    clippy::non_canonical_clone_impl,
    clippy::non_canonical_partial_ord_impl,
    clippy::needless_lifetimes,
    dead_code
)]
#![warn(clippy::expl_impl_clone_on_copy)]


#[derive(Copy)]
struct Qux;

impl Clone for Qux {
    //~^ ERROR: you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self {
        Qux
    }
}

// looks like unions don't support deriving Clone for now
#[derive(Copy)]
union Union {
    a: u8,
}

impl Clone for Union {
    fn clone(&self) -> Self {
        Union { a: 42 }
    }
}

// See #666
#[derive(Copy)]
struct Lt<'a> {
    a: &'a u8,
}

impl<'a> Clone for Lt<'a> {
    //~^ ERROR: you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Copy)]
struct BigArray {
    a: [u8; 65],
}

impl Clone for BigArray {
    //~^ ERROR: you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Copy)]
struct FnPtr {
    a: fn() -> !,
}

impl Clone for FnPtr {
    //~^ ERROR: you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

// Ok, Clone trait impl doesn't have constrained generics.
#[derive(Copy)]
struct Generic<T> {
    a: T,
}

impl<T> Clone for Generic<T> {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Copy)]
struct Generic2<T>(T);
impl<T: Clone> Clone for Generic2<T> {
    //~^ ERROR: you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// Ok, Clone trait impl doesn't have constrained generics.
#[derive(Copy)]
struct GenericRef<'a, T, U>(T, &'a U);
impl<T: Clone, U> Clone for GenericRef<'_, T, U> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }
}

// https://github.com/rust-lang/rust-clippy/issues/10188
#[repr(packed)]
#[derive(Copy)]
struct Packed<T>(T);

impl<T: Copy> Clone for Packed<T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn main() {}
