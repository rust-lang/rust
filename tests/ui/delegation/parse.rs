//@ check-pass

#![feature(decl_macro)]
#![feature(fn_delegation)]
#![allow(incomplete_features)]

macro_rules! reuse { {} => {} }

mod reuse {
    pub fn to_unsafe(x: i32) -> i32 { x + 1 }
    pub fn to_pub() {}
    pub fn to_pub2() {}

    mod inner {
        #[allow(non_camel_case_types)]
        struct reuse {
            a: i32,
            b: i32,
            c: i32,
        }

        impl reuse {
            reuse!();
        }

        fn baz() {
            let (a, b, c) = (0, 0, 0);
            reuse {a, b, c};
        }
    }

    pub macro my_macro() {}
}

reuse!();
reuse::my_macro!();

#[inline]
pub reuse reuse::to_pub;
pub reuse crate::reuse::to_pub2;

fn main() {}
