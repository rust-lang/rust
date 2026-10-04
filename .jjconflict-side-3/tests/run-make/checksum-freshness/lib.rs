// A basic library to be used in tests with no real purpose.

mod foo;
// Binary file with invalid UTF-8 sequence.
static BINARY_FILE: &[u8] = include_bytes!("binary_file");
pub fn sum(a: i32, b: i32) -> i32 {
    a + b
}
