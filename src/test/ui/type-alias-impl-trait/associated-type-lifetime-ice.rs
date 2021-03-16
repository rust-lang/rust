// failure-status: 101
// rustc-env:RUST_BACKTRACE=0
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""

// compile-flags: --crate-type=rlib

// Regression test for https://github.com/rust-lang/rust/issues/78450

#![feature(min_type_alias_impl_trait)]
#![no_std]

pub trait AssociatedImpl {
    type ImplTrait;

    fn f() -> Self::ImplTrait;
}

struct S<T>(T);

trait Associated {
    type A;
}

// ICE
impl<'a, T: Associated<A = &'a ()>> AssociatedImpl for S<T> {
    type ImplTrait = impl core::fmt::Debug;

    fn f() -> Self::ImplTrait {
    //~^ ERROR unexpected concrete region in borrowck: ReEarlyBound(0, 'a)
        ()
    }
}
