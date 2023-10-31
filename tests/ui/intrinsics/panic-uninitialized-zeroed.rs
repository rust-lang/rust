// run-pass
// revisions: default strict
// [strict]compile-flags: -Zstrict-init-checks
// ignore-tidy-linelength
// ignore-emscripten spawning processes is not supported
// ignore-sgx no processes

// This test checks panic emitted from `mem::{uninitialized,zeroed}`.

#![feature(never_type)]
#![allow(deprecated, invalid_value)]

use std::{
    mem::{self, MaybeUninit, ManuallyDrop},
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

#[allow(dead_code, non_camel_case_types)]
enum OneVariant_Ref {
    Variant(&'static i32),
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

#[track_caller]
fn test_panic_msg<T, F: (FnOnce() -> T) + 'static>(op: F, msg: &str) {
    use std::{panic, env, process};

    // The tricky part is that we can't just run `op`, as that would *abort* the process.
    // So instead, we reinvoke this process with the caller location as argument.
    // For the purpose of this test, the line number is unique enough.
    // If we are running in such a re-invocation, we skip all the tests *except* for the one with that type name.
    let our_loc = panic::Location::caller().line().to_string();
    let mut args = env::args();
    let this = args.next().unwrap();
    if let Some(loc) = args.next() {
        if loc == our_loc {
            op();
            panic!("we did not abort");
        } else {
            // Nothing, we are running another test.
        }
    } else {
        // Invoke new process for actual test, and check result.
        let mut cmd = process::Command::new(this);
        cmd.arg(our_loc);
        let res = cmd.output().unwrap();
        assert!(!res.status.success(), "test did not fail");
        let stderr = String::from_utf8_lossy(&res.stderr);
        assert!(stderr.contains(msg), "test did not contain expected output: looking for {:?}, output:\n{}", msg, stderr);
    }
}

#[track_caller]
fn test_panic_msg_only_if_strict<T>(op: impl (FnOnce() -> T) + 'static, msg: &str) {
    if !cfg!(strict) {
        // Just run it.
        op();
    } else {
        test_panic_msg(op, msg);
    }
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
            || mem::uninitialized::<[Bar; 2]>(),
            "attempted to instantiate uninhabited type `[Bar; 2]`"
        );
        test_panic_msg(
            || mem::zeroed::<[Bar; 2]>(),
            "attempted to instantiate uninhabited type `[Bar; 2]`"
        );
        test_panic_msg(
            || MaybeUninit::<[Bar; 2]>::uninit().assume_init(),
            "attempted to instantiate uninhabited type `[Bar; 2]`"
        );

        // Types that don't allow either.
        test_panic_msg(
            || mem::zeroed::<&i32>(),
            "attempted to zero-initialize type `&i32`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<&i32>(),
            "attempted to leave type `&i32` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<Box<[i32; 0]>>(),
            "attempted to zero-initialize type `alloc::boxed::Box<[i32; 0]>`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<Box<[i32; 0]>>(),
            "attempted to leave type `alloc::boxed::Box<[i32; 0]>` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<Box<u8>>(),
            "attempted to zero-initialize type `alloc::boxed::Box<u8>`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<Box<u8>>(),
            "attempted to leave type `alloc::boxed::Box<u8>` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&[i32]>(),
            "attempted to zero-initialize type `&[i32]`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<&[i32]>(),
            "attempted to leave type `&[i32]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&(u8, [u8])>(),
            "attempted to zero-initialize type `&(u8, [u8])`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<&(u8, [u8])>(),
            "attempted to leave type `&(u8, [u8])` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&dyn Send>(),
            "attempted to zero-initialize type `&dyn core::marker::Send`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<&dyn Send>(),
            "attempted to leave type `&dyn core::marker::Send` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<*const dyn Send>(),
            "attempted to zero-initialize type `*const dyn core::marker::Send`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<*const dyn Send>(),
            "attempted to leave type `*const dyn core::marker::Send` uninitialized, which is invalid"
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
            || mem::zeroed::<OneVariant_Ref>(),
            "attempted to zero-initialize type `OneVariant_Ref`, \
                which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<OneVariant_Ref>(),
            "attempted to leave type `OneVariant_Ref` uninitialized, which is invalid"
        );

        // Types where both are invalid, but we allow uninit since the 0x01-filling is not LLVM UB.
        test_panic_msg(
            || mem::zeroed::<fn()>(),
            "attempted to zero-initialize type `fn()`, which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<fn()>(),
            "attempted to leave type `fn()` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&()>(),
            "attempted to zero-initialize type `&()`, which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<&()>(),
            "attempted to leave type `&()` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&[u8]>(),
            "attempted to zero-initialize type `&[u8]`, which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<&[u8]>(),
            "attempted to leave type `&[u8]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<&str>(),
            "attempted to zero-initialize type `&str`, which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<&str>(),
            "attempted to leave type `&str` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<(NonNull<u32>, u32, u32)>(),
            "attempted to zero-initialize type `(core::ptr::non_null::NonNull<u32>, u32, u32)`, \
                which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<(NonNull<u32>, u32, u32)>(),
            "attempted to leave type `(core::ptr::non_null::NonNull<u32>, u32, u32)` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<OneVariant_NonZero>(),
            "attempted to zero-initialize type `OneVariant_NonZero`, \
                which is invalid"
        );
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<OneVariant_NonZero>(),
            "attempted to leave type `OneVariant_NonZero` uninitialized, which is invalid"
        );

        // Types where both are invalid but we allow the zeroed form since it is not LLVM UB.
        test_panic_msg_only_if_strict(
            || mem::zeroed::<LR_NonZero>(),
            "attempted to zero-initialize type `LR_NonZero`, which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<LR_NonZero>(),
            "attempted to leave type `LR_NonZero` uninitialized, which is invalid"
        );

        test_panic_msg_only_if_strict(
            || mem::zeroed::<ManuallyDrop<LR_NonZero>>(),
            "attempted to zero-initialize type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>`, \
             which is invalid"
        );
        test_panic_msg(
            || mem::uninitialized::<ManuallyDrop<LR_NonZero>>(),
            "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>` uninitialized, \
             which is invalid"
        );

        // Some strict-only things
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<i32>(),
            "attempted to leave type `i32` uninitialized, which is invalid"
        );

        test_panic_msg_only_if_strict(
            || mem::uninitialized::<*const ()>(),
            "attempted to leave type `*const ()` uninitialized, which is invalid"
        );

        test_panic_msg_only_if_strict(
            || mem::uninitialized::<[i32; 1]>(),
            "attempted to leave type `[i32; 1]` uninitialized, which is invalid"
        );

        test_panic_msg_only_if_strict(
            || mem::zeroed::<[NonNull<()>; 1]>(),
            "attempted to zero-initialize type `[core::ptr::non_null::NonNull<()>; 1]`, which is invalid"
        );

        // Types that can be zero, but not uninit (though some are mitigated).
        let _val = mem::zeroed::<LR>();
        test_panic_msg(
            || mem::uninitialized::<LR>(),
            "attempted to leave type `LR` uninitialized, which is invalid"
        );

        let _val = mem::zeroed::<ManuallyDrop<LR>>();
        test_panic_msg(
            || mem::uninitialized::<ManuallyDrop<LR>>(),
            "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR>` uninitialized, which is invalid"
        );

        let _val = mem::zeroed::<bool>();
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<bool>(),
            "attempted to leave type `bool` uninitialized, which is invalid"
        );

        let _val = mem::zeroed::<OneVariant>();
        test_panic_msg_only_if_strict(
            || mem::uninitialized::<OneVariant>(),
            "attempted to leave type `OneVariant` uninitialized, which is invalid"
        );

        // Some things that are actually allowed.
        let _val = mem::zeroed::<Option<&'static i32>>();
        let _val = mem::zeroed::<MaybeUninit<NonNull<u32>>>();
        let _val = mem::zeroed::<[!; 0]>();
        let _val = mem::zeroed::<ZeroIsValid>();
        let _val = mem::uninitialized::<MaybeUninit<bool>>();
        let _val = mem::uninitialized::<[!; 0]>();
        let _val: () = mem::uninitialized::<()>();
        let _val = mem::uninitialized::<ZeroSized>();
    }
}
