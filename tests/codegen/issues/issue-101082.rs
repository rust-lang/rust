//@ compile-flags: -Copt-level=3
//@ revisions: host x86-64 x86-64-v3
//@ min-llvm-version: 20

//@[host] ignore-x86_64

// Set the base cpu explicitly, in case the default has been changed.
//@[x86-64] only-x86_64
//@[x86-64] compile-flags: -Ctarget-cpu=x86-64

// FIXME(cuviper) x86-64-v3 in particular regressed in #131563, and the
// workaround at the time still sometimes fails, so ignore it for now.
// - https://github.com/llvm/llvm-project/issues/134513
// - https://github.com/llvm/llvm-project/issues/134735
//@[x86-64-v3] ignore-test
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
