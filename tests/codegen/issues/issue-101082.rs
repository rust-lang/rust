//@ compile-flags: -Copt-level=3
//@ revisions: host x86-64-v3
//@ min-llvm-version: 20

// This particular CPU regressed in #131563
//@[x86-64-v3] only-x86_64
//@[x86-64-v3] compile-flags: -Ctarget-cpu=x86-64-v3

#![crate_type = "lib"]

#[no_mangle]
pub fn test() -> usize {
    // CHECK-LABEL: @test(
    // CHECK: ret {{i64|i32}} 165
    let values = [23, 16, 54, 3, 60, 9];
    let mut acc = 0;
    for item in values {
        acc += item;
    }
    acc
}
