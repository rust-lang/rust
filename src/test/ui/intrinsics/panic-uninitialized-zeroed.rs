// run-pass
// needs-unwind
// ignore-wasm32-bare compiled with panic=abort by default
// revisions: mir thir strict
// [thir]compile-flags: -Zthir-unsafeck
// [strict]compile-flags: -Zstrict-init-checks
// ignore-tidy-linelength

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

struct ZeroSized;

#[allow(dead_code)]
#[repr(i32)]
enum ZeroIsValid {
    Zero(u8) = 0,
    One(NonNull<()>) = 1,
}

#[rustfmt::skip]
#[allow(dead_code)]
enum SoManyVariants {
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17,
    A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32,
    A33, A34, A35, A36, A37, A38, A39, A40, A41, A42, A43, A44, A45, A46, A47,
    A48, A49, A50, A51, A52, A53, A54, A55, A56, A57, A58, A59, A60, A61, A62,
    A63, A64, A65, A66, A67, A68, A69, A70, A71, A72, A73, A74, A75, A76, A77,
    A78, A79, A80, A81, A82, A83, A84, A85, A86, A87, A88, A89, A90, A91, A92,
    A93, A94, A95, A96, A97, A98, A99, A100, A101, A102, A103, A104, A105, A106,
    A107, A108, A109, A110, A111, A112, A113, A114, A115, A116, A117, A118, A119,
    A120, A121, A122, A123, A124, A125, A126, A127, A128, A129, A130, A131, A132,
    A133, A134, A135, A136, A137, A138, A139, A140, A141, A142, A143, A144, A145,
    A146, A147, A148, A149, A150, A151, A152, A153, A154, A155, A156, A157, A158,
    A159, A160, A161, A162, A163, A164, A165, A166, A167, A168, A169, A170, A171,
    A172, A173, A174, A175, A176, A177, A178, A179, A180, A181, A182, A183, A184,
    A185, A186, A187, A188, A189, A190, A191, A192, A193, A194, A195, A196, A197,
    A198, A199, A200, A201, A202, A203, A204, A205, A206, A207, A208, A209, A210,
    A211, A212, A213, A214, A215, A216, A217, A218, A219, A220, A221, A222, A223,
    A224, A225, A226, A227, A228, A229, A230, A231, A232, A233, A234, A235, A236,
    A237, A238, A239, A240, A241, A242, A243, A244, A245, A246, A247, A248, A249,
    A250, A251, A252, A253, A254, A255, A256,
}

#[track_caller]
fn test_panic_msg<T>(op: impl (FnOnce() -> T) + panic::UnwindSafe, msg: &str) {
    let err = panic::catch_unwind(op).err();
    assert_eq!(
        err.as_ref().and_then(|a| a.downcast_ref::<&str>()),
        Some(&msg)
    );
}

#[track_caller]
// If strict mode is enabled, expect the msg. Otherwise, expect there to be no error.
fn test_strict_panic_msg<T>(op: impl (FnOnce() -> T) + panic::UnwindSafe, msg: &str) {
    let err = panic::catch_unwind(op).err();

    let expectation = if cfg!(strict) { Some(&msg) } else { None };

    assert_eq!(
        err.as_ref().and_then(|a| a.downcast_ref::<&str>()),
        expectation
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
            "attempted to leave type `*const dyn core::marker::Send` uninitialized, which is invalid"
        );
        test_panic_msg(
            || mem::zeroed::<*const dyn Send>(),
            "attempted to zero-initialize type `*const dyn core::marker::Send`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<(NonNull<u32>, u32, u32)>(),
            "attempted to leave type `(core::ptr::non_null::NonNull<u32>, u32, u32)` uninitialized, \
                which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<(NonNull<u32>, u32, u32)>(),
            "attempted to zero-initialize type `(core::ptr::non_null::NonNull<u32>, u32, u32)`, \
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
            || mem::uninitialized::<LR_NonZero>(),
            "attempted to leave type `LR_NonZero` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<ManuallyDrop<LR_NonZero>>(),
            "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>` uninitialized, \
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
            "attempted to leave type `core::mem::manually_drop::ManuallyDrop<LR>` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<&'static [u8]>(),
            "attempted to leave type `&[u8]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<&'static [u16]>(),
            "attempted to leave type `&[u16]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<SoManyVariants>(),
            "attempted to leave type `SoManyVariants` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<[(&'static [u8], &'static str); 1]>(),
            "attempted to zero-initialize type `[(&[u8], &str); 1]`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<[&'static [u16]; 1]>(),
            "attempted to leave type `[&[u16]; 1]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<[NonNull<()>; 1]>(),
            "attempted to zero-initialize type `[core::ptr::non_null::NonNull<()>; 1]`, which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<[NonNull<()>; 1]>(),
            "attempted to leave type `[core::ptr::non_null::NonNull<()>; 1]` uninitialized, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<LR_NonZero>(),
            "attempted to zero-initialize type `LR_NonZero`, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<[LR_NonZero; 1]>(),
            "attempted to zero-initialize type `[LR_NonZero; 1]`, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<[LR_NonZero; 1]>(),
            "attempted to zero-initialize type `[LR_NonZero; 1]`, which is invalid"
        );

        test_panic_msg(
            || mem::zeroed::<ManuallyDrop<LR_NonZero>>(),
            "attempted to zero-initialize type `core::mem::manually_drop::ManuallyDrop<LR_NonZero>`, \
             which is invalid"
        );

        test_panic_msg(
            || mem::uninitialized::<(&'static [u8], &'static str)>(),
            "attempted to leave type `(&[u8], &str)` uninitialized, which is invalid"
        );

        // Some things that should work.
        let _val = mem::zeroed::<bool>();
        let _val = mem::zeroed::<LR>();
        let _val = mem::zeroed::<ManuallyDrop<LR>>();
        let _val = mem::zeroed::<OneVariant>();
        let _val = mem::zeroed::<Option<&'static i32>>();
        let _val = mem::zeroed::<MaybeUninit<NonNull<u32>>>();
        let _val = mem::zeroed::<[!; 0]>();
        let _val = mem::zeroed::<ZeroIsValid>();
        let _val = mem::zeroed::<SoManyVariants>();
        let _val = mem::uninitialized::<[SoManyVariants; 0]>();
        let _val = mem::uninitialized::<MaybeUninit<bool>>();
        let _val = mem::uninitialized::<[!; 0]>();
        let _val = mem::uninitialized::<()>();
        let _val = mem::uninitialized::<ZeroSized>();

        // These are UB because they have not been officially blessed, but we await the resolution
        // of <https://github.com/rust-lang/unsafe-code-guidelines/issues/71> before doing
        // anything about that.
        test_strict_panic_msg(
            || mem::uninitialized::<i32>(),
            "attempted to leave type `i32` uninitialized, which is invalid"
        );

        test_strict_panic_msg(
            || mem::uninitialized::<*const ()>(),
            "attempted to leave type `*const ()` uninitialized, which is invalid"
        );

        test_strict_panic_msg(
            || mem::uninitialized::<[i32; 1]>(),
            "attempted to leave type `[i32; 1]` uninitialized, which is invalid"
        );

        test_strict_panic_msg(
            || mem::uninitialized::<[(&'static [u8], &'static str); 1]>(),
            "attempted to leave type `[(&[u8], &str); 1]` uninitialized, which is invalid"
        );
    }
}
