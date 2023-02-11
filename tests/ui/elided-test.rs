// error-pattern: `main` function not found

// Since we're not compiling a test runner this function should be elided
// and the build will fail because main doesn't exist
#[test]
fn main() {
}
