#![warn(clippy::fn_params_excessive_bools)]
#![allow(clippy::too_many_arguments)]

extern "C" {
    // Should not lint, most of the time users have no control over extern function signatures
    fn f(_: bool, _: bool, _: bool, _: bool);
}

macro_rules! foo {
    () => {
        fn fff(_: bool, _: bool, _: bool, _: bool) {}
    };
}

foo!();

#[no_mangle]
extern "C" fn k(_: bool, _: bool, _: bool, _: bool) {}
fn g(_: bool, _: bool, _: bool, _: bool) {}
fn h(_: bool, _: bool, _: bool) {}
fn e(_: S, _: S, _: Box<S>, _: Vec<u32>) {}
fn t(_: S, _: S, _: Box<S>, _: Vec<u32>, _: bool, _: bool, _: bool, _: bool) {}

struct S;
trait Trait {
    // should warn for trait functions with and without body
    fn f(_: bool, _: bool, _: bool, _: bool);
    fn g(_: bool, _: bool, _: bool, _: Vec<u32>);
    #[allow(clippy::fn_params_excessive_bools)]
    fn h(_: bool, _: bool, _: bool, _: bool, _: bool, _: bool);
    fn i(_: bool, _: bool, _: bool, _: bool) {}
}

impl S {
    fn f(&self, _: bool, _: bool, _: bool, _: bool) {}
    fn g(&self, _: bool, _: bool, _: bool) {}
    #[no_mangle]
    extern "C" fn h(_: bool, _: bool, _: bool, _: bool) {}
}

impl Trait for S {
    // Should not lint because the trait might not be changeable by the user
    // We only lint in the trait definition
    fn f(_: bool, _: bool, _: bool, _: bool) {}
    fn g(_: bool, _: bool, _: bool, _: Vec<u32>) {}
    fn h(_: bool, _: bool, _: bool, _: bool, _: bool, _: bool) {}
}

fn main() {
    fn n(_: bool, _: u32, _: bool, _: Box<u32>, _: bool, _: bool) {
        fn nn(_: bool, _: bool, _: bool, _: bool) {}
    }
}
