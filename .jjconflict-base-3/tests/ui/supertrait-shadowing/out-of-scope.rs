//@ run-pass

#![feature(gca_min_const_items)]
#![allow(dead_code)]

use std::mem::size_of;

mod out_of_scope {
    use std::gca;
    pub trait Subtrait: super::Supertrait {
        fn hello(&self) -> &'static str {
            "subtrait"
        }
        type Assoc;
        #[rustc_always_gca]
        const CONST: i32;
    }
    impl<T> Subtrait for T {
        type Assoc = i16;
        const CONST: i32 = gca!(2);
    }
}

trait Supertrait {
    fn hello(&self) -> &'static str {
        "supertrait"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> Supertrait for T {
    type Assoc = i8;
    const CONST: i32 = 1;
}

fn main() {
    assert_eq!(().hello(), "supertrait");
    check::<()>();
}

fn check<T: Supertrait>() {
    assert_eq!(size_of::<T::Assoc>(), 1);
    assert_eq!(T::CONST, 1);
}
