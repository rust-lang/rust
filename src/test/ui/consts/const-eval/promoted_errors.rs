// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-pass
// ignore-pass (test emits codegen-time warnings and verifies that they are not errors)

#![warn(const_err, arithmetic_overflow, unconditional_panic)]

// The only way to have promoteds that fail is in `const fn` called from `const`/`static`.
const fn overflow() -> u32 {
    0 - 1
    //[opt_with_overflow_checks,noopt]~^ WARN any use of this value will cause an error
    //[opt_with_overflow_checks,noopt]~| WARN this was previously accepted by the compiler
}
const fn div_by_zero1() -> i32 {
    1 / 0
    //[opt]~^ WARN any use of this value will cause an error
    //[opt]~| WARN this was previously accepted by the compiler but is being phased out
}
const fn div_by_zero2() -> i32 {
    1 / (1 - 1)
}
const fn div_by_zero3() -> i32 {
    1 / (false as i32)
}
const fn oob() -> i32 {
    [1, 2, 3][4]
}

const X: () = {
    let _x: &'static u32 = &overflow();
    //[opt_with_overflow_checks,noopt]~^ WARN any use of this value will cause an error
    //[opt_with_overflow_checks,noopt]~| WARN this was previously accepted by the compiler
    let _x: &'static i32 = &div_by_zero1();
    //[opt]~^ WARN any use of this value will cause an error
    //[opt]~| WARN this was previously accepted by the compiler but is being phased out
    let _x: &'static i32 = &div_by_zero2();
    let _x: &'static i32 = &div_by_zero3();
    let _x: &'static i32 = &oob();
};

fn main() {}
