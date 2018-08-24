// error-pattern:panicked at 'test-fail-fmt 42 rust'

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
