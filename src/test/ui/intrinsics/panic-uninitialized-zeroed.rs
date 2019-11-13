// run-pass
// ignore-wasm32-bare compiled with panic=abort by default

// This test checks panic emitted from `mem::{uninitialized,zeroed}`.

#![feature(never_type)]
#![allow(deprecated, invalid_value)]

use std::{
    mem::{self, MaybeUninit},
    panic,
    ptr::NonNull,
};

#[allow(dead_code)]
struct Foo {
    x: u8,
    y: !,
}

enum Bar {}

#[allow(dead_code)]
enum OneVariant { Variant(i32) }

fn test_panic_msg<T>(op: impl (FnOnce() -> T) + panic::UnwindSafe, msg: &str) {
    let err = panic::catch_unwind(op).err();
    assert_eq!(
        err.as_ref().and_then(|a| a.downcast_ref::<String>()).map(|s| &**s),
        Some(msg)
    );
}

fn main() {
    unsafe {
        // Uninitialized types
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

        // Types that do not like zero-initialziation
        test_panic_msg(
            || mem::uninitialized::<fn()>(),
            "attempted to leave type `fn()` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<fn()>(),
            "attempted to zero-initialize type `fn()`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<*const dyn Send>(),
            "attempted to leave type `*const dyn std::marker::Send` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<*const dyn Send>(),
            "attempted to zero-initialize type `*const dyn std::marker::Send`, which is invalid"
        );

        /* FIXME(#66151) we conservatively do not error here yet.
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
        */

        test_panic_msg(
            || mem::uninitialized::<bool>(),
            "attempted to leave type `bool` uninitialized, which is invalid"
        );

        // Some things that should work.
        let _val = mem::zeroed::<bool>();
        let _val = mem::zeroed::<OneVariant>();
        let _val = mem::zeroed::<Option<&'static i32>>();
        let _val = mem::zeroed::<MaybeUninit<NonNull<u32>>>();
        let _val = mem::uninitialized::<MaybeUninit<bool>>();

        // We don't panic for these just to be conservative. They are UB as of now (2019-11-09).
        let _val = mem::uninitialized::<i32>();
        let _val = mem::uninitialized::<*const ()>();
    }
}
