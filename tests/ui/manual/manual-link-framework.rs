//@ignore-target-macos
//@ignore-target-ios
//@compile-flags:-l framework=foo
//@error-in-other-file: library kind `framework` is only supported on Apple targets

fn main() {
}
