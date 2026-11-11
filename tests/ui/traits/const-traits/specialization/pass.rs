//@ check-pass
#![feature(const_trait_impl, const_default, min_specialization, rustc_attrs)]
#![allow(internal_features)]

#[rustc_specialization_trait]
pub const unsafe trait Sup {
    fn foo() -> u32;
}

#[rustc_specialization_trait]
pub const unsafe trait Sub: [const] Sup {}

const unsafe impl Sup for u8 {
    default fn foo() -> u32 {
        1
    }
}

const unsafe impl Sup for () {
    fn foo() -> u32 {
        42
    }
}

const unsafe impl Sub for () {}

pub const trait A {
    fn a() -> u32;
}

const impl<T: [const] Default> A for T {
    default fn a() -> u32 {
        2
    }
}

const impl<T: [const] Default + [const] Sup> A for T {
    default fn a() -> u32 {
        3
    }
}

const impl<T: [const] Default + [const] Sub> A for T {
    fn a() -> u32 {
        T::foo()
    }
}

const _: () = assert!(<()>::a() == 42);
const _: () = assert!(<u8>::a() == 3);
const _: () = assert!(<u16>::a() == 2);

fn main() {}
