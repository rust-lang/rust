// Ensure `memchr` communicates that a returned index is in bounds.
//@ compile-flags: -Copt-level=3 -Zinline-mir=false
//@ only-64bit

#![crate_type = "lib"]

// CHECK-LABEL: @find_char
#[no_mangle]
pub fn find_char(haystack: &str, needle: char) -> Option<usize> {
    // CHECK-NOT: phi { i64, i64 }
    // CHECK: ret { i64, i64 }
    haystack.find(needle)
}
