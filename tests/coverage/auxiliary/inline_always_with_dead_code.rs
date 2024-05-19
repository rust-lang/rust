//@ compile-flags: -Cinstrument-coverage -Ccodegen-units=4 -Copt-level=0

#![allow(dead_code)]

mod foo {
    #[inline(always)]
    pub fn called() {}

    fn uncalled() {}
}

pub mod bar {
    pub fn call_me() {
        super::foo::called();
    }
}

pub mod baz {
    pub fn call_me() {
        super::foo::called();
    }
}
