// compile-flags: --env FOO=123abc -Zunstable-options
// run-pass
fn main() {
    assert_eq!(env!("FOO"), "123abc");
}
