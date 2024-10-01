// Test that inherent str constructors cannot be used when inherent_str_constructors
// feature gate is not used.

fn main() {
    str::from_utf8(b"hi").unwrap();
}
