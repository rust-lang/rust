// error-pattern:assertion failed: `(left != right)`
// error-pattern: left: `14`
// error-pattern:right: `14`

fn main() {
    assert_ne!(14, 14);
}
