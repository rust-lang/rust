// check-fail
// compile-flags: -Zverbose

#![feature(generic_associated_types)]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
pub trait Copy {}

pub trait Functor {
    type With<T>;

    fn fmap<T, U>(this: Self::With<T>) -> Self::With<U>;
}

pub trait FunctorExt<T>: Sized {
    type Base: Functor<With<T> = Self>;

    fn fmap<U>(self) {
        let arg: <Self::Base as Functor>::With<T>;
        let ret: <Self::Base as Functor>::With<U>;

        arg = self;
        ret = <Self::Base as Functor>::fmap(arg);
        //~^ mismatched types
    }
}
