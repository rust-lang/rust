//@ run-pass
// Tests that Option<E> niche optimisation does not incorrectly use i32::MIN
// as the None niche when a repr(i32) enum variant already holds that value.
// github.com/rust-lang/rust/issues/49973
#[derive(Debug)]
#[repr(i32)]
enum E {
    Min = -2147483648i32,
    _Max = 2147483647i32,
}

fn main() {
    assert_eq!(Some(E::Min).unwrap() as i32, -2147483648i32);
}
