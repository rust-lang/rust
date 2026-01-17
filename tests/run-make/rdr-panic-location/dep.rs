// Dependency crate that can panic at known locations.
// The panic locations are used to verify that -Z stable-crate-hash
// preserves correct file:line:col information.

#![crate_type = "rlib"]
#![crate_name = "dep"]

/// Public function that panics. The panic location should be
/// correctly reported even when compiled with -Z stable-crate-hash.
#[inline(never)]
pub fn will_panic(trigger: bool) {
    if trigger {
        panic!("intentional panic for testing"); // Line 13
    }
}

/// Public function with panic in a private helper.
/// Tests that spans in private code are still correctly resolved.
#[inline(never)]
pub fn panic_via_private(trigger: bool) {
    private_panicker(trigger);
}

#[inline(never)]
fn private_panicker(trigger: bool) {
    if trigger {
        panic!("panic from private function"); // Line 27
    }
}

/// Panic with a formatted message to test span handling
/// in format string expansion.
#[inline(never)]
pub fn panic_with_format(value: i32) {
    if value < 0 {
        panic!("invalid value: {}", value); // Line 36
    }
}
