//@ rustc-env: FOO=123abc
//@ run-pass
fn main() {
    assert_eq!(env!("FOO"), "123abc");
}
