// ignore-linux
// ignore-macos

// Test that panics on Windows give a reasonable error message.

// error-pattern: panicking is not supported on this target
fn main() {
    core::panic!("this is {}", "Windows");
}
