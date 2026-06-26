// This is a regression test for https://github.com/rust-lang/rust/issues/147265.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn mk_result(a: usize) -> Result<u8, *const u8> {
    // CHECK-LABEL: @mk_result
    // CHECK-NOT: unreachable
    // CHECK: load i8,
    // CHECK-NOT: unreachable
    match g(a) {
        Ok(b) => Ok(unsafe { *(b as *const u8) }),
        Err(c) => Err(c),
    }
}

#[cold]
fn g(a: usize) -> Result<usize, *const u8> {
    Ok(a)
}
