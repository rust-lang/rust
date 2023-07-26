//! This test checks that we don't lose hidden types
//! for *other* opaque types that we register and use
//! to prove bounds while checking that a hidden type
//! satisfies its opaque type's bounds.

#![feature(trivial_bounds, type_alias_impl_trait)]
#![allow(trivial_bounds)]

mod sus {
    use super::*;
    pub type Sep = impl Sized + std::fmt::Display;
    //~^ ERROR: concrete type differs from previous defining opaque type use
    pub fn mk_sep() -> Sep {
        String::from("hello")
    }

    pub trait Proj {
        type Assoc;
    }
    impl Proj for () {
        type Assoc = sus::Sep;
    }

    pub struct Bar<T: Proj> {
        pub inner: <T as Proj>::Assoc,
        pub _marker: T,
    }
    impl<T: Proj> Clone for Bar<T> {
        fn clone(&self) -> Self {
            todo!()
        }
    }
    impl<T: Proj<Assoc = i32> + Copy> Copy for Bar<T> {}
    // This allows producing `Tait`s via `From`, even though
    // `define_tait` is not actually callable, and thus assumed
    // `Bar<()>: Copy` even though it isn't.
    pub type Tait = impl Copy + From<Bar<()>> + Into<Bar<()>>;
    pub fn define_tait() -> Tait
    where
        // this proves `Bar<()>: Copy`, but `define_tait` is
        // now uncallable
        (): Proj<Assoc = i32>,
    {
        Bar { inner: 1i32, _marker: () }
    }
}

fn copy_tait(x: sus::Tait) -> (sus::Tait, sus::Tait) {
    (x, x)
}

fn main() {
    let bar = sus::Bar { inner: sus::mk_sep(), _marker: () };
    let (y, z) = copy_tait(bar.into()); // copy a string
    drop(y.into()); // drop one instance
    println!("{}", z.into().inner); // print the other
}
