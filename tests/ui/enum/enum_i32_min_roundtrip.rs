//@ run-pass
//Tests enum with full i-32 range discriminants roundtrip correctly through option and casting.
//https://github.com/rust-lang/rust/issues/49973
#[derive(Debug)]
#[repr(i32)]
enum E {
    Min = -2147483648i32,
    _Max = 2147483647i32,
}

fn main() {
    assert_eq!(Some(E::Min).unwrap() as i32, -2147483648i32);
}
