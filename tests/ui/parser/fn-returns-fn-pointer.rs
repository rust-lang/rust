// check-pass
// Regression test for #78507.
fn foo() -> Option<fn() -> Option<bool>> {
    Some(|| Some(true))
}
fn main() {}
