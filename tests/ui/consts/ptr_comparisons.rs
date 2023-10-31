// compile-flags: --crate-type=lib
// check-pass

#![feature(
    core_intrinsics,
    const_raw_ptr_comparison,
)]

const FOO: &usize = &42;

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

// We want pointers to be equal to themselves, but aren't checking this yet because
// there are some open questions (e.g. whether function pointers to the same function
// compare equal: they don't necessarily do at runtime).
check!(!, FOO as *const _, FOO as *const _);

// aside from 0, these pointers might end up pretty much anywhere.
check!(!, FOO as *const _, 1); // this one could be `ne` by taking into account alignment
check!(!, FOO as *const _, 1024);

// When pointers go out-of-bounds, they *might* become null, so these comparions cannot work.
check!(!, unsafe { (FOO as *const usize).wrapping_add(2) }, 0);
check!(!, unsafe { (FOO as *const usize).wrapping_sub(1) }, 0);
