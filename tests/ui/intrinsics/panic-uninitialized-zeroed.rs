// run-pass
// needs-unwind
// ignore-wasm32-bare compiled with panic=abort by default
// revisions: mir thir strict
// [thir]compile-flags: -Zthir-unsafeck
// [strict]compile-flags: -Zstrict-init-checks
// ignore-tidy-linelength

// This test checks panic emitted from `mem::{uninitialized,zeroed}` and `MaybeUninit::{uninit,zeroed}::assume_init`.

#![feature(never_type, arbitrary_enum_discriminant)]
#![allow(deprecated, invalid_value)]

use std::{
    mem::{uninitialized as mem_uninit, zeroed as mem_zeroed, ManuallyDrop, MaybeUninit},
    num, panic,
    ptr::NonNull,
};

#[allow(dead_code)]
struct Foo {
    x: u8,
    y: !,
}

enum Bar {}

#[allow(dead_code)]
enum OneVariant {
    Variant(i32),
}

#[allow(dead_code, non_camel_case_types)]
enum OneVariant_NonZero {
    Variant(i32, i32, num::NonZeroI32),
    DeadVariant(Bar),
}

// An `Aggregate` abi enum where 0 is not a valid discriminant.
#[allow(dead_code)]
#[repr(i32)]
enum NoNullVariant {
    Variant1(i32, i32) = 1,
    Variant2(i32, i32) = 2,
}

// An enum with ScalarPair layout
#[allow(dead_code)]
enum LR {
    Left(i64),
    Right(i64),
}
#[allow(dead_code, non_camel_case_types)]
enum LR_NonZero {
    Left(num::NonZeroI64),
    Right(num::NonZeroI64),
}

struct ZeroSized;

#[allow(dead_code)]
#[repr(i32)]
enum ZeroIsValid {
    Zero(u8) = 0,
    One(NonNull<()>) = 1,
}

fn test_panic_msg<T>(op: impl (FnOnce() -> T) + panic::UnwindSafe, msg: &str) {
    let err = panic::catch_unwind(op).err();
    assert_eq!(err.as_ref().and_then(|a| a.downcast_ref::<&str>()), Some(&msg));
}

macro_rules! test_zero_uninit {
    ($zero_fn:ident, $uninit_fn:ident) => {
        unsafe {
            // Uninhabited types
            test_panic_msg(
                || $uninit_fn::<!>(),
                "attempted to instantiate uninhabited type `!`"
            );
            test_panic_msg(
                || $zero_fn::<!>(),
                "attempted to instantiate uninhabited type `!`"
            );

            test_panic_msg(
                || $uninit_fn::<Foo>(),
                "attempted to instantiate uninhabited type `Foo`"
            );
            test_panic_msg(
                || $zero_fn::<Foo>(),
                "attempted to instantiate uninhabited type `Foo`"
            );

            test_panic_msg(
                || $uninit_fn::<Bar>(),
                "attempted to instantiate uninhabited type `Bar`"
            );
            test_panic_msg(
                || $zero_fn::<Bar>(),
                "attempted to instantiate uninhabited type `Bar`"
            );

            test_panic_msg(
                || $uninit_fn::<[Foo; 2]>(),
                "attempted to instantiate uninhabited type `[Foo; 2]`"
            );
            test_panic_msg(
                || $zero_fn::<[Foo; 2]>(),
                "attempted to instantiate uninhabited type `[Foo; 2]`"
            );

            test_panic_msg(
                || $uninit_fn::<[Bar; 2]>(),
                "attempted to instantiate uninhabited type `[Bar; 2]`"
            );
            test_panic_msg(
                || $zero_fn::<[Bar; 2]>(),
                "attempted to instantiate uninhabited type `[Bar; 2]`"
            );

            // Types that do not like zero-initialziation
            test_panic_msg(
                || $uninit_fn::<fn()>(),
                "attempted to leave type `fn()` uninitialized, which is invalid"
            );
            test_panic_msg(
                || $zero_fn::<fn()>(),
                "attempted to zero-initialize type `fn()`, which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<*const dyn Send>(),
                "attempted to leave type `*const dyn core::marker::Send` uninitialized, which is invalid"
            );
            test_panic_msg(
                || $zero_fn::<*const dyn Send>(),
                "attempted to zero-initialize type `*const dyn core::marker::Send`, which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<(NonNull<u32>, u32, u32)>(),
                "attempted to leave type `(core::ptr::non_null::NonNull<u32>, u32, u32)` uninitialized, \
                    which is invalid"
            );

            test_panic_msg(
                || $zero_fn::<(NonNull<u32>, u32, u32)>(),
                "attempted to zero-initialize type `(core::ptr::non_null::NonNull<u32>, u32, u32)`, \
                    which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<OneVariant_NonZero>(),
                "attempted to leave type `OneVariant_NonZero` uninitialized, \
                    which is invalid"
            );
            test_panic_msg(
                || $zero_fn::<OneVariant_NonZero>(),
                "attempted to zero-initialize type `OneVariant_NonZero`, \
                    which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<LR_NonZero>(),
                "attempted to leave type `LR_NonZero` uninitialized, which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<ManuallyDrop<LR_NonZero>>(),
                "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>` uninitialized, \
                 which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<NoNullVariant>(),
                "attempted to leave type `NoNullVariant` uninitialized, \
                    which is invalid"
            );

            test_panic_msg(
                || $zero_fn::<NoNullVariant>(),
                "attempted to zero-initialize type `NoNullVariant`, \
                    which is invalid"
            );

            // Types that can be zero, but not uninit.
            test_panic_msg(
                || $uninit_fn::<bool>(),
                "attempted to leave type `bool` uninitialized, which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<LR>(),
                "attempted to leave type `LR` uninitialized, which is invalid"
            );

            test_panic_msg(
                || $uninit_fn::<ManuallyDrop<LR>>(),
                "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR>` uninitialized, which is invalid"
            );

            // Some things that should work.
            let _val = $zero_fn::<bool>();
            let _val = $zero_fn::<LR>();
            let _val = $zero_fn::<ManuallyDrop<LR>>();
            let _val = $zero_fn::<OneVariant>();
            let _val = $zero_fn::<Option<&'static i32>>();
            let _val = $zero_fn::<MaybeUninit<NonNull<u32>>>();
            let _val = $zero_fn::<[!; 0]>();
            let _val = $zero_fn::<ZeroIsValid>();
            let _val = $uninit_fn::<MaybeUninit<bool>>();
            let _val = $uninit_fn::<[!; 0]>();
            let _val = $uninit_fn::<()>();
            let _val = $uninit_fn::<ZeroSized>();

            if cfg!(strict) {
                test_panic_msg(
                    || $uninit_fn::<i32>(),
                    "attempted to leave type `i32` uninitialized, which is invalid"
                );

                test_panic_msg(
                    || $uninit_fn::<*const ()>(),
                    "attempted to leave type `*const ()` uninitialized, which is invalid"
                );

                test_panic_msg(
                    || $uninit_fn::<[i32; 1]>(),
                    "attempted to leave type `[i32; 1]` uninitialized, which is invalid"
                );

                test_panic_msg(
                    || $zero_fn::<NonNull<()>>(),
                    "attempted to zero-initialize type `core::ptr::non_null::NonNull<()>`, which is invalid"
                );

                test_panic_msg(
                    || $zero_fn::<[NonNull<()>; 1]>(),
                    "attempted to zero-initialize type `[core::ptr::non_null::NonNull<()>; 1]`, which is invalid"
                );

                // FIXME(#66151) we conservatively do not error here yet (by default).
                test_panic_msg(
                    || $zero_fn::<LR_NonZero>(),
                    "attempted to zero-initialize type `LR_NonZero`, which is invalid"
                );

                test_panic_msg(
                    || $zero_fn::<ManuallyDrop<LR_NonZero>>(),
                    "attempted to zero-initialize type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>`, \
                     which is invalid"
                );
            } else {
                // These are UB because they have not been officially blessed, but we await the resolution
                // of <https://github.com/rust-lang/unsafe-code-guidelines/issues/71> before doing
                // anything about that.
                let _val = $uninit_fn::<i32>();
                let _val = $uninit_fn::<*const ()>();

                // These are UB, but best to test them to ensure we don't become unintentionally
                // stricter.

                // It's currently unchecked to create invalid enums and values inside arrays.
                let _val = $zero_fn::<LR_NonZero>();
                let _val = $zero_fn::<[LR_NonZero; 1]>();
                let _val = $zero_fn::<[NonNull<()>; 1]>();
                let _val = $uninit_fn::<[NonNull<()>; 1]>();
            }
        }
    }
}

unsafe fn maybe_uninit_uninit<T>() -> T {
    MaybeUninit::uninit().assume_init()
}

unsafe fn maybe_uninit_zeroed<T>() -> T {
    MaybeUninit::zeroed().assume_init()
}

fn main() {
    test_zero_uninit!(mem_zeroed, mem_uninit);
    test_zero_uninit!(maybe_uninit_zeroed, maybe_uninit_uninit);
}
