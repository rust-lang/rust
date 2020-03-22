// ignore-linux
// ignore-macos

// Test that panics on Windows give a reasonable error message.

// error-pattern: panicking is not supported on this target
#[allow(unconditional_panic)]
fn main() {
    let _val = 1/0;
}
