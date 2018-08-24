// error-pattern:assertion failed: `(left == right)`
// error-pattern: left: `14`
// error-pattern:right: `15`

fn main() {
    assert_eq!(14, 15);
}
