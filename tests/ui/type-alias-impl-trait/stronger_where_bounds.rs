// known-bug: #113278
// run-fail

#![feature(trivial_bounds, type_alias_impl_trait)]
#![allow(trivial_bounds)]

mod sus {
    use super::*;
    pub type Sep = impl Sized + std::fmt::Display;
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
    pub type Tait = impl Copy + From<Bar<()>> + Into<Bar<()>>;
    pub fn define_tait() -> Tait
    where
        // This bound does not exist on `type Tait`, but will
        // constrain `Sep` to `i32`, and then forget about it.
        // On the other hand `mk_sep` constrains it to `String`.
        // Since `Tait: Copy`, we can now copy `String`s.
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
