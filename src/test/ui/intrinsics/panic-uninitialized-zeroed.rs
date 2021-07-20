// run-pass
// ignore-wasm32-bare compiled with panic=abort by default
// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

// This test checks panic emitted from `mem::{uninitialized,zeroed}`.

#![feature(never_type, arbitrary_enum_discriminant)]
#![allow(deprecated, invalid_value)]

use std::{
    mem::{self, MaybeUninit, ManuallyDrop},
    panic,
    ptr::NonNull,
    num,
};

#[allow(dead_code)]
struct Foo {
    x: u8,
    y: !,
}

enum Bar {}

#[allow(dead_code)]
enum OneVariant { Variant(i32) }

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

fn test_panic_msg<T>(op: impl (FnOnce() -> T) + panic::UnwindSafe, msg: &str) {
    let err = panic::catch_unwind(op).err();
    assert_eq!(
        err.as_ref().and_then(|a| a.downcast_ref::<&str>()),
        Some(&msg)
    );
}

fn main() {
    unsafe {
        // Uninhabited types
        test_panic_msg(
            || mem::uninitialized::<!>(),
            "attempted to instantiate uninhabited type `!`"
        );
        test_panic_msg(
            || mem::zeroed::<!>(),
            "attempted to instantiate uninhabited type `!`"
        );
        test_panic_msg(
            || MaybeUninit::<!>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `!`"
        );
        test_panic_msg(
            || mem::uninitialized::<[!; 2]>(),
            "attempted to instantiate uninhabited type `[!; 2]`"
        );
        test_panic_msg(
            || mem::zeroed::<[!; 2]>(),
            "attempted to instantiate uninhabited type `[!; 2]`"
        );
        test_panic_msg(
            || MaybeUninit::<[!; 2]>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `[!; 2]`"
        );

        test_panic_msg(
            || mem::uninitialized::<Foo>(),
            "attempted to instantiate uninhabited type `Foo`"
        );
        test_panic_msg(
            || mem::zeroed::<Foo>(),
            "attempted to instantiate uninhabited type `Foo`"
        );
        test_panic_msg(
            || MaybeUninit::<Foo>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `Foo`"
        );
        test_panic_msg(
            || mem::uninitialized::<[Foo; 2]>(),
            "attempted to instantiate uninhabited type `[Foo; 2]`"
        );
        test_panic_msg(
            || mem::zeroed::<[Foo; 2]>(),
            "attempted to instantiate uninhabited type `[Foo; 2]`"
        );
        test_panic_msg(
            || MaybeUninit::<[Foo; 2]>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `[Foo; 2]`"
        );

        test_panic_msg(
            || mem::uninitialized::<Bar>(),
            "attempted to instantiate uninhabited type `Bar`"
        );
        test_panic_msg(
            || mem::zeroed::<Bar>(),
            "attempted to instantiate uninhabited type `Bar`"
        );
        test_panic_msg(
            || MaybeUninit::<Bar>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `Bar`"
        );
        test_panic_msg(
            || mem::uninitialized::<[Bar; 4]>(),
            "attempted to instantiate uninhabited type `[Bar; 4]`"
        );
        test_panic_msg(
            || mem::zeroed::<[Bar; 4]>(),
            "attempted to instantiate uninhabited type `[Bar; 4]`"
        );
        test_panic_msg(
            || MaybeUninit::<[Bar; 4]>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `[Bar; 4]`"
        );

        // Types that do not like zero-initialization
        test_panic_msg(
            || mem::uninitialized::<fn()>(),
            "attempted to leave type `fn()` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<fn()>(),
            "attempted to zero-initialize type `fn()`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[fn(); 2]>(),
            "attempted to leave type `[fn(); 2]` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<[fn(); 2]>(),
            "attempted to zero-initialize type `[fn(); 2]`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<*const dyn Send>(),
            "attempted to leave type `*const dyn std::marker::Send` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<*const dyn Send>(),
            "attempted to zero-initialize type `*const dyn std::marker::Send`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[*const dyn Send; 2]>(),
            "attempted to leave type `[*const dyn std::marker::Send; 2]` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<[*const dyn Send; 2]>(),
            "attempted to zero-initialize type `[*const dyn std::marker::Send; 2]`, \
                which is invalid"
        );

        /* FIXME(#66151) we conservatively do not error here yet.
        test_panic_msg(
            || mem::uninitialized::<LR_NonZero>(),
            "attempted to leave type `LR_NonZero` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<LR_NonZero>(),
            "attempted to zero-initialize type `LR_NonZero`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<ManuallyDrop<LR_NonZero>>(),
            "attempted to leave type `std::mem::ManuallyDrop<LR_NonZero>` uninitialized, \
             which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<ManuallyDrop<LR_NonZero>>(),
            "attempted to zero-initialize type `std::mem::ManuallyDrop<LR_NonZero>`, \
             which is invalid"
        );
        */

        test_panic_msg(
            || mem::uninitialized::<(NonNull<u32>, u32, u32)>(),
            "attempted to leave type `(std::ptr::NonNull<u32>, u32, u32)` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<(NonNull<u32>, u32, u32)>(),
            "attempted to zero-initialize type `(std::ptr::NonNull<u32>, u32, u32)`, \
                which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[(NonNull<u32>, u32, u32); 2]>(),
            "attempted to leave type `[(std::ptr::NonNull<u32>, u32, u32); 2]` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<[(NonNull<u32>, u32, u32); 2]>(),
            "attempted to zero-initialize type `[(std::ptr::NonNull<u32>, u32, u32); 2]`, \
                which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<OneVariant_NonZero>(),
            "attempted to leave type `OneVariant_NonZero` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<OneVariant_NonZero>(),
            "attempted to zero-initialize type `OneVariant_NonZero`, \
                which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[OneVariant_NonZero; 2]>(),
            "attempted to leave type `[OneVariant_NonZero; 2]` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<[OneVariant_NonZero; 2]>(),
            "attempted to zero-initialize type `[OneVariant_NonZero; 2]`, \
                which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<NoNullVariant>(),
            "attempted to leave type `NoNullVariant` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<NoNullVariant>(),
            "attempted to zero-initialize type `NoNullVariant`, \
                which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[NoNullVariant; 2]>(),
            "attempted to leave type `[NoNullVariant; 2]` uninitialized, \
                which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<[NoNullVariant; 2]>(),
            "attempted to zero-initialize type `[NoNullVariant; 2]`, \
                which is invalid"
        );

        // Types that can be zero, but not uninit.
        test_panic_msg(
            || mem::uninitialized::<bool>(),
            "attempted to leave type `bool` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<LR>(),
            "attempted to leave type `LR` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<ManuallyDrop<LR>>(),
            "attempted to leave type `std::mem::ManuallyDrop<LR>` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[bool; 2]>(),
            "attempted to leave type `[bool; 2]` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[LR; 2]>(),
            "attempted to leave type `[LR; 2]` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<[ManuallyDrop<LR>; 2]>(),
            "attempted to leave type `[std::mem::ManuallyDrop<LR>; 2]` uninitialized, \
                which is invalid"
        );

        // Some things that should work.
        let _val = mem::zeroed::<bool>();
        let _val = mem::zeroed::<[bool; 4]>();
        let _val = mem::zeroed::<LR>();
        let _val = mem::zeroed::<[LR; 8]>();
        let _val = mem::zeroed::<ManuallyDrop<LR>>();
        let _val = mem::zeroed::<[ManuallyDrop<LR>; 16]>();
        let _val = mem::zeroed::<OneVariant>();
        let _val = mem::zeroed::<[OneVariant; 2]>();
        let _val = mem::zeroed::<Option<&'static i32>>();
        let _val = mem::zeroed::<[Option<&'static i32>; 3]>();
        let _val = mem::zeroed::<MaybeUninit<NonNull<u32>>>();
        let _val = mem::zeroed::<[MaybeUninit<NonNull<u32>>; 32]>();
        let _val = mem::zeroed::<[!; 0]>();
        let _val = mem::uninitialized::<MaybeUninit<bool>>();
        let _val = mem::uninitialized::<[MaybeUninit<bool>; 1]>();
        let _val = mem::uninitialized::<[bool; 0]>();
        let _val = mem::uninitialized::<[!; 0]>();
        let _val = mem::uninitialized::<[fn(); 0]>();
        let _val = mem::uninitialized::<[*const dyn Send; 0]>();

        // These are UB because they have not been officially blessed, but we await the resolution
        // of <https://github.com/rust-lang/unsafe-code-guidelines/issues/71> before doing
        // anything about that.
        let _val = mem::uninitialized::<i32>();
        let _val = mem::uninitialized::<[i32; 1]>();
        let _val = mem::uninitialized::<*const ()>();
        let _val = mem::uninitialized::<[*const (); 2]>();
    }
}
