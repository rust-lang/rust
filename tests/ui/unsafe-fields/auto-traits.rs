//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(auto_traits)]
#![feature(unsafe_fields)]
#![allow(incomplete_features)]

enum UnsafeEnum {
    Safe(u8),
    Unsafe { unsafe field: u8 },
}

auto trait SafeAuto {}

fn impl_safe_auto(_: impl SafeAuto) {}

unsafe auto trait UnsafeAuto {}

fn impl_unsafe_auto(_: impl UnsafeAuto) {}

fn tests() {
    impl_safe_auto(UnsafeEnum::Safe(42));
    impl_unsafe_auto(UnsafeEnum::Safe(42));
    //~^ ERROR the trait bound `UnsafeEnum: UnsafeAuto` is not satisfied
}
