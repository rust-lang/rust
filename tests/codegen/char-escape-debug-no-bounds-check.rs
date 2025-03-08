//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::char::EscapeDebug;

// Make sure no bounds checks are emitted when escaping a character.

// CHECK-LABEL: @char_escape_debug_no_bounds_check
#[no_mangle]
pub fn char_escape_debug_no_bounds_check(c: char) -> EscapeDebug {
    // CHECK-NOT: panic
    // CHECK-NOT: panic_bounds_check
    c.escape_debug()
}
