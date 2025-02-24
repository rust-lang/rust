// Checks that these functions are branchless.
//
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @is_ascii_alphanumeric_char
#[no_mangle]
pub fn is_ascii_alphanumeric_char(x: char) -> bool {
    // CHECK-NOT: br
    x.is_ascii_alphanumeric()
}

// CHECK-LABEL: @is_ascii_alphanumeric_u8
#[no_mangle]
pub fn is_ascii_alphanumeric_u8(x: u8) -> bool {
    // CHECK-NOT: br
    x.is_ascii_alphanumeric()
}

// CHECK-LABEL: @is_ascii_hexdigit_char
#[no_mangle]
pub fn is_ascii_hexdigit_char(x: char) -> bool {
    // CHECK-NOT: br
    x.is_ascii_hexdigit()
}

// CHECK-LABEL: @is_ascii_hexdigit_u8
#[no_mangle]
pub fn is_ascii_hexdigit_u8(x: u8) -> bool {
    // CHECK-NOT: br
    x.is_ascii_hexdigit()
}

// CHECK-LABEL: @is_ascii_punctuation_char
#[no_mangle]
pub fn is_ascii_punctuation_char(x: char) -> bool {
    // CHECK-NOT: br
    x.is_ascii_punctuation()
}

// CHECK-LABEL: @is_ascii_punctuation_u8
#[no_mangle]
pub fn is_ascii_punctuation_u8(x: u8) -> bool {
    // CHECK-NOT: br
    x.is_ascii_punctuation()
}
