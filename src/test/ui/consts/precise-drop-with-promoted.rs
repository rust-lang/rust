// Regression test for issue #89938.
// check-pass
// compile-flags: --crate-type=lib
#![feature(const_precise_live_drops)]

pub const fn f() {
    let _: Option<String> = None;
    let _: &'static Option<String> = &None;
}
