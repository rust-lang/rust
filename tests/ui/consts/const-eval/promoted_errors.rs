// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-pass
// ignore-pass (test emits codegen-time warnings and verifies that they are not errors)

//! This test ensures that when we promote code that fails to evaluate, the build still succeeds.

#![warn(arithmetic_overflow, unconditional_panic)]

// The only way to have promoteds that fail is in `const fn` called from `const`/`static`.
const fn overflow() -> u32 {
    0 - 1
    //~^ WARN this arithmetic operation will overflow
}
const fn div_by_zero1() -> i32 {
    1 / 0
    //~^ WARN this operation will panic at runtime
}
const fn div_by_zero2() -> i32 {
    1 / (1 - 1)
    //~^ WARN this operation will panic at runtime
}
const fn div_by_zero3() -> i32 {
    1 / (false as i32)
    //~^ WARN this operation will panic at runtime
}
const fn oob() -> i32 {
    [1, 2, 3][4]
    //~^ WARN this operation will panic at runtime
}

const fn mk_false() -> bool { false }

// An actually used constant referencing failing promoteds in dead code.
// This needs to always work.
const Y: () = {
    if mk_false() {
        let _x: &'static u32 = &overflow();
        let _x: &'static i32 = &div_by_zero1();
        let _x: &'static i32 = &div_by_zero2();
        let _x: &'static i32 = &div_by_zero3();
        let _x: &'static i32 = &oob();
    }
    ()
};

fn main() {
    Y;
}
