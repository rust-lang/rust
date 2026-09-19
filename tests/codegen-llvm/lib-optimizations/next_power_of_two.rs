//@ compile-flags: -C opt-level=3 -Z merge-functions=disabled
//@ only-64bit (avoid worrying about `usize` width)
//@ revisions: aarch64 x86_64
//@ [aarch64] only-aarch64
//@ [x86_64] only-x86_64
//@ [x86_64] compile-flags: -C target-cpu=x86-64-v3
// (The codegen comes out differently in -v1 so use -v3 to make it the same as aarch64.)

#![crate_type = "lib"]

// The idea behind the current implementation is that by returning `1 << …`
// LLVM can see that it's obviously a power of two, which wasn't the case
// in the implementation that did `(-1 >> …) + 1`.

#[unsafe(no_mangle)]
pub fn full_npot(x: u64) -> u64 {
    // CHECK-LABEL: @full_npot
    // CHECK: [[M1:%.+]] = tail call i64 @llvm.usub.sat.i64(i64 %x, i64 1)
    // CHECK: [[LZ:%.+]] = tail call {{.+}} i64 @llvm.ctlz.i64(i64 [[M1]], i1 false)
    // CHECK: [[BW:%.+]] = sub nuw nsw i64 64, [[LZ]]
    // CHECK: [[POT:%.+]] = shl nuw i64 1, [[BW]]
    // CHECK: [[IS_WRAPPING:%.+]] = icmp slt i64 [[M1]], 0
    // CHECK: [[RET:%.+]] = select i1 [[IS_WRAPPING]], i64 0, i64 [[POT]]
    // CHECK: ret i64 [[RET]]
    x.next_power_of_two()
}

// With a restricted input, both edge cases optimize away
#[unsafe(no_mangle)]
pub unsafe fn restricted_npot(x: u16) -> u16 {
    std::hint::assert_unchecked(1 <= x);
    // largest value that can get `shl nsw` too.
    std::hint::assert_unchecked(x <= 0x4000);

    // CHECK-LABEL: @restricted_npot(
    // CHECK: [[M1:%.+]] = add nsw i16 %x, -1
    // CHECK: [[LZ:%.+]] = tail call {{.+}} i16 @llvm.ctlz.i16(i16 [[M1]], i1 false)
    // CHECK: [[BW:%.+]] = sub nuw nsw i16 16, [[LZ]]
    // CHECK: [[P2:%.+]] = shl nuw nsw i16 1, [[BW]]
    // CHECK: ret i16 [[P2]]
    (x as u16).next_power_of_two()
}

// Slices (of non-ZSTs) are short enough that the power-of-two always fits
#[unsafe(no_mangle)]
pub fn slice_length_npot(slice: &[u8]) -> usize {
    // CHECK-LABEL: @slice_length_npot(
    // CHECK: [[POT:%.+]] = shl nuw i64 1,
    // CHECK: ret i64 [[POT]]
    slice.len().next_power_of_two()
}

#[unsafe(no_mangle)]
pub fn checked_npot_is_pot(x: u32) -> bool {
    // CHECK-LABEL: @checked_npot_is_pot
    // CHECK: ret i1 true
    x.checked_next_power_of_two().is_none_or(u32::is_power_of_two)
}

#[unsafe(no_mangle)]
pub fn wrapping_npot_is_pot_or_zero(x: u32) -> bool {
    // CHECK-LABEL: @wrapping_npot_is_pot_or_zero
    // CHECK: ret i1 true
    x.next_power_of_two().count_ones() <= 1
}

// Because it knows it's a power of two, it can rewrite modulo to masking
#[no_mangle]
pub fn modulo_npot(value: u64, dividend: u64) {
    // CHECK-LABEL: @modulo_npot
    // CHECK: [[HIGHBITS:%.+]] = shl nsw i64 -1,
    // CHECK: [[LOWBITS:%.+]] = xor i64 [[HIGHBITS]], -1
    // CHECK: [[REMAINDER:%.+]] = and i64 %dividend, [[LOWBITS]]
    // CHECK: @do_something(i64 noundef [[REMAINDER]])
    if let Some(pot) = value.checked_next_power_of_two() {
        do_something(dividend % pot)
    }
}

unsafe extern "Rust" {
    safe fn do_something(_: u64);
}
