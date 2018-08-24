// error-pattern:panicked at 'test-assert-owned'

fn main() {
    assert!(false, "test-assert-owned".to_string());
}
