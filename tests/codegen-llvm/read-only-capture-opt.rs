//@ compile-flags: -C opt-level=3 -Z mir-opt-level=0
//@ min-llvm-version: 21

#![crate_type = "lib"]

unsafe extern "C" {
    safe fn do_something(p: &i32);
}

#[unsafe(no_mangle)]
pub fn test() -> i32 {
    // CHECK-LABEL: @test(
    // CHECK: ret i32 0
    let i = 0;
    do_something(&i);
    do_something(&i);
    i
}
