// compile-flags: --crate-type=lib
// normalize-stderr-32bit: "offset 8" -> "offset $$TWO_WORDS"
// normalize-stderr-64bit: "offset 16" -> "offset $$TWO_WORDS"
// normalize-stderr-32bit: "size 4" -> "size $$WORD"
// normalize-stderr-64bit: "size 8" -> "size $$WORD"

#![feature(
    const_panic,
    core_intrinsics,
    const_raw_ptr_comparison,
    const_ptr_offset,
    const_raw_ptr_deref,
    raw_ref_macros
)]

const FOO: &usize = &42;

macro_rules! check {
    (eq, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(std::intrinsics::ptr_guaranteed_eq($a as *const u8, $b as *const u8));
    };
    (ne, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(std::intrinsics::ptr_guaranteed_ne($a as *const u8, $b as *const u8));
    };
    (!eq, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(!std::intrinsics::ptr_guaranteed_eq($a as *const u8, $b as *const u8));
    };
    (!ne, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(!std::intrinsics::ptr_guaranteed_ne($a as *const u8, $b as *const u8));
    };
}

check!(eq, 0, 0);
check!(ne, 0, 1);
check!(!eq, 0, 1);
check!(!ne, 0, 0);
check!(ne, FOO as *const _, 0);
check!(!eq, FOO as *const _, 0);
// We want pointers to be equal to themselves, but aren't checking this yet because
// there are some open questions (e.g. whether function pointers to the same function
// compare equal, they don't necessarily at runtime).
// The case tested here should work eventually, but does not work yet.
check!(!eq, FOO as *const _, FOO as *const _);
check!(ne, unsafe { (FOO as *const usize).offset(1) }, 0);
check!(!eq, unsafe { (FOO as *const usize).offset(1) }, 0);

check!(ne, unsafe { (FOO as *const usize as *const u8).offset(3) }, 0);
check!(!eq, unsafe { (FOO as *const usize as *const u8).offset(3) }, 0);

///////////////////////////////////////////////////////////////////////////////
// If any of the below start compiling, make sure to add a `check` test for it.
// These invocations exist as canaries so we don't forget to check that the
// behaviour of `guaranteed_eq` and `guaranteed_ne` is still correct.
// All of these try to obtain an out of bounds pointer in some manner. If we
// can create out of bounds pointers, we can offset a pointer far enough that
// at runtime it would be zero and at compile-time it would not be zero.

const _: *const usize = unsafe { (FOO as *const usize).offset(2) };
//~^ NOTE

const _: *const u8 =
//~^ NOTE
    unsafe { std::ptr::raw_const!((*(FOO as *const usize as *const [u8; 1000]))[999]) };
//~^ ERROR any use of this value will cause an error
//~| NOTE

const _: usize = unsafe { std::mem::transmute::<*const usize, usize>(FOO) + 4 };
//~^ ERROR any use of this value will cause an error
//~| NOTE "pointer-to-integer cast" needs an rfc
//~| NOTE

const _: usize = unsafe { *std::mem::transmute::<&&usize, &usize>(&FOO) + 4 };
//~^ ERROR any use of this value will cause an error
//~| NOTE "pointer-to-integer cast" needs an rfc
//~| NOTE
