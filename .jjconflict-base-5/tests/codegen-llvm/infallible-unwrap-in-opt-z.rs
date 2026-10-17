//@ compile-flags: -C opt-level=z
//@ edition: 2021

#![crate_type = "lib"]

// From <https://github.com/rust-lang/rust/issues/115463>

// CHECK-LABEL: @read_up_to_8(
#[no_mangle]
pub fn read_up_to_8(buf: &[u8]) -> u64 {
    // CHECK-NOT: unwrap_failed
    if buf.len() < 4 {
        // actual instance has more code.
        return 0;
    }
    let lo = u32::from_le_bytes(buf[..4].try_into().unwrap()) as u64;
    let hi = u32::from_le_bytes(buf[buf.len() - 4..][..4].try_into().unwrap()) as u64;
    lo | (hi << 8 * (buf.len() as u64 - 4))
}

// CHECK-LABEL: @checking_unwrap_expectation(
#[no_mangle]
pub fn checking_unwrap_expectation(buf: &[u8]) -> &[u8; 4] {
    // CHECK: call void @{{.*core6result13unwrap_failed}}
    buf.try_into().unwrap()
}
