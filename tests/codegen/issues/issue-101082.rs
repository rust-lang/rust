//@ compile-flags: -Copt-level=3
//@ revisions: host x86-64 x86-64-v3
//@ min-llvm-version: 20

//@[host] ignore-x86_64

// Set the base cpu explicitly, in case the default has been changed.
//@[x86-64] only-x86_64
//@[x86-64] compile-flags: -Ctarget-cpu=x86-64

// FIXME(cuviper) x86-64-v3 in particular regressed in #131563, and the workaround
// at the time still sometimes fails, so only verify it for the power-of-two size
// - https://github.com/llvm/llvm-project/issues/134735
//@[x86-64-v3] only-x86_64
//@[x86-64-v3] compile-flags: -Ctarget-cpu=x86-64-v3

#![crate_type = "lib"]

#[no_mangle]
pub fn test() -> usize {
    // CHECK-LABEL: @test(
    // host: ret {{i64|i32}} 165
    // x86-64: ret {{i64|i32}} 165

    // FIXME: Now that this autovectorizes via a masked load, it doesn't actually
    // const-fold for certain widths.  The `test_eight` case below shows that, yes,
    // what we're emitting *can* be const-folded, except that the way LLVM does it
    // for certain widths doesn't today.  We should be able to put this back to
    // the same check after <https://github.com/llvm/llvm-project/issues/134513>
    // x86-64-v3: masked.load

    let values = [23, 16, 54, 3, 60, 9];
    let mut acc = 0;
    for item in values {
        acc += item;
    }
    acc
}

#[no_mangle]
pub fn test_eight() -> usize {
    // CHECK-LABEL: @test_eight(
    // CHECK: ret {{i64|i32}} 220
    let values = [23, 16, 54, 3, 60, 9, 13, 42];
    let mut acc = 0;
    for item in values {
        acc += item;
    }
    acc
}
