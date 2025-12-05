// https://github.com/rust-lang/rust/issues/85461
//@ compile-flags: -Cinstrument-coverage -Ccodegen-units=4 --crate-type dylib -Copt-level=0
//@ build-pass
//@ needs-profiler-runtime
//@ needs-dynamic-linking

// Regression test for #85461 where MSVC sometimes fails to link instrument-coverage binaries
// with dead code and #[inline(always)].

#![allow(dead_code)]

mod foo {
    #[inline(always)]
    pub fn called() { }

    fn uncalled() { }
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
