// This test case checks that LLVM is aware that computing the size of a slice cannot wrap.
// The possibility of wrapping results in an additional branch when dropping boxed slices
// in some situations, see https://github.com/rust-lang/rust/issues/96497#issuecomment-1112865218

//@ compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @simple_size_of_nowrap
#[no_mangle]
pub fn simple_size_of_nowrap(x: &[u32]) -> usize {
    // Make sure the shift used to compute the size has a nowrap flag.

    // CHECK: [[A:%.*]] = shl nsw {{.*}}, 2
    // CHECK-NEXT: ret {{.*}} [[A]]
    core::mem::size_of_val(x)
}

// CHECK-LABEL: @drop_write
#[no_mangle]
pub fn drop_write(mut x: Box<[u32]>) {
    // Check that this write is optimized out.
    // This depends on the size calculation not wrapping,
    // since otherwise LLVM can't tell that the memory is always deallocated if the slice len > 0.

    // CHECK-NOT: store i32 42
    x[1] = 42;
}
