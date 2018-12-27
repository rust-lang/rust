// error-pattern:panicked at 'test-assert-fmt 42 rust'

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
