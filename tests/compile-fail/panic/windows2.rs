// ignore-linux
// ignore-macos

// Test that panics on Windows give a reasonable error message.

// error-pattern: panicking is not supported on this target
fn main() {
    std::panic!("this is Windows");
}
