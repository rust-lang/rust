//@ known-bug: #110395

#![feature(const_trait_impl, min_specialization, rustc_attrs)]

use std::fmt::Debug;

#[rustc_specialization_trait]
#[const_trait]
pub unsafe trait Sup {
    fn foo() -> u32;
}

#[rustc_specialization_trait]
#[const_trait]
pub unsafe trait Sub: [const] Sup {}

unsafe impl const Sup for u8 {
    default fn foo() -> u32 {
        1
    }
}

unsafe impl const Sup for () {
    fn foo() -> u32 {
        42
    }
}

unsafe impl const Sub for () {}

#[const_trait]
pub trait A {
    fn a() -> u32;
}

impl<T: [const] Debug> const A for T {
    default fn a() -> u32 {
        2
    }
}

impl<T: [const] Debug + [const] Sup> const A for T {
    default fn a() -> u32 {
        3
    }
}

impl<T: [const] Debug + [const] Sub> const A for T {
    fn a() -> u32 {
        T::foo()
    }
}

const _: () = assert!(<()>::a() == 42);
const _: () = assert!(<u8>::a() == 3);
const _: () = assert!(<u16>::a() == 2);

fn main() {}
