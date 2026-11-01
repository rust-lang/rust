//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Make sure no char boundary checks are emitted when slicing or splitting after
// a byte known to be an ASCII character.

// CHECK-LABEL: @slice_after_ascii
#[no_mangle]
pub fn slice_after_ascii(s: &str) -> (&str, &str) {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_error_fail
    let mut mid = s.len();
    let v = s.as_bytes();
    for index in 0..v.len() {
        if v[index] == b'\n' {
            mid = index + 1;
            break;
        }
    }
    (&s[..mid], &s[mid..])
}

// CHECK-LABEL: @try_slice_after_ascii
#[no_mangle]
pub fn try_slice_after_ascii(s: &str) -> Option<(&str, &str)> {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_error_fail
    let v = s.as_bytes();
    for index in 0..v.len() {
        if v[index] == b'\n' {
            let mid = index + 1;
            return Some((&s[..mid], &s[mid..]));
        }
    }
    None
}

// CHECK-LABEL: @split_after_ascii
#[no_mangle]
pub fn split_after_ascii(s: &str) -> (&str, &str) {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_error_fail
    let mut mid = s.len();
    let v = s.as_bytes();
    for index in 0..v.len() {
        if v[index] == b'\n' {
            mid = index + 1;
            break;
        }
    }
    s.split_at(mid)
}

// CHECK-LABEL: @try_split_after_ascii
#[no_mangle]
pub fn try_split_after_ascii(s: &str) -> Option<(&str, &str)> {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_error_fail
    let v = s.as_bytes();
    for index in 0..v.len() {
        if v[index] == b'\n' {
            return Some(s.split_at(index + 1));
        }
    }
    None
}
