//@ check-pass

#![feature(decl_macro)]

macro m($T:ident, $f:ident) {
    pub trait $T {
        fn f(&self) -> u32 { 0 }
        fn $f(&self) -> i32 { 0 }
    }
    impl $T for () {}

    let _: u32 = ().f();
    let _: i32 = ().$f();
}

fn main() {
    m!(T, f);
    let _: i32 = ().f();
}
