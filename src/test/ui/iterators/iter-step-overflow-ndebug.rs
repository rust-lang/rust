// run-pass
// compile-flags: -C debug_assertions=no

fn main() {
    let mut it = u8::max_value()..;
    assert_eq!(it.next().unwrap(), 255);
    assert_eq!(it.next().unwrap(), u8::min_value());

    let mut it = i8::max_value()..;
    assert_eq!(it.next().unwrap(), 127);
    assert_eq!(it.next().unwrap(), i8::min_value());
}
