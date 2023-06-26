// compile-flags: --crate-type=lib
// normalize-stderr-32bit: "8 bytes" -> "$$TWO_WORDS bytes"
// normalize-stderr-64bit: "16 bytes" -> "$$TWO_WORDS bytes"
// normalize-stderr-32bit: "size 4" -> "size $$WORD"
// normalize-stderr-64bit: "size 8" -> "size $$WORD"

#![feature(
    core_intrinsics,
    const_raw_ptr_comparison,
)]

const FOO: &usize = &42;
const BAR: &usize = &43;

fn foo() {}
fn bar() {}

macro_rules! check {
    (eq, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(std::intrinsics::ptr_guaranteed_cmp($a as *const u8, $b as *const u8) == 1);
    };
    (ne, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(std::intrinsics::ptr_guaranteed_cmp($a as *const u8, $b as *const u8) == 0);
    };
    (!, $a:expr, $b:expr) => {
        pub const _: () =
            assert!(std::intrinsics::ptr_guaranteed_cmp($a as *const u8, $b as *const u8) == 2);
    };
}

check!(eq, 0, 0);
check!(ne, 0, 1);
check!(ne, FOO as *const _, 0);
check!(ne, unsafe { (FOO as *const usize).offset(1) }, 0);
check!(ne, unsafe { (FOO as *const usize as *const u8).offset(3) }, 0);
check!(eq, FOO as *const _, FOO as *const _);
check!(ne, unsafe { (FOO as *const usize).offset(1) }, FOO as *const _);

// Comparisons of pointers to different allocations cannot be computed in the
// general case, but some more cases could be implemented eventually:

// This could be known, because FOO points to different data than BAR, so the
// two pointers must be unequal.
check!(!, FOO as *const _, BAR as *const _);
// Function pointer addresses could overlap with the addresses of one-past-end-pointers
// so this check would need to be extra precise.
check!(!, foo as *const usize, FOO as *const _);

// This comparison cannot be known until the whole memory layout is,
// so it cannot be performed in CTFE.
check!(!, unsafe { (FOO as *const usize).offset(1) }, BAR as *const _);
// Function pointer comparions are flaky even at runtime.
check!(!, foo as *const usize, bar as *const usize);

///////////////////////////////////////////////////////////////////////////////
// If any of the below start compiling, make sure to add a `check` test for it.
// These invocations exist as canaries so we don't forget to check that the
// behaviour of `guaranteed_eq` and `guaranteed_ne` is still correct.
// All of these try to obtain an out of bounds pointer in some manner. If we
// can create out of bounds pointers, we can offset a pointer far enough that
// at runtime it would be zero and at compile-time it would not be zero.

const _: *const usize = unsafe { (FOO as *const usize).offset(2) };

const _: *const u8 =
    unsafe { std::ptr::addr_of!((*(FOO as *const usize as *const [u8; 1000]))[999]) };
//~^ ERROR evaluation of constant value failed
//~| out-of-bounds

const _: usize = unsafe { std::mem::transmute::<*const usize, usize>(FOO) + 4 };
//~^ ERROR evaluation of constant value failed
//~| unable to turn pointer into raw bytes

const _: usize = unsafe { *std::mem::transmute::<&&usize, &usize>(&FOO) + 4 };
//~^ ERROR evaluation of constant value failed
//~| unable to turn pointer into raw bytes
