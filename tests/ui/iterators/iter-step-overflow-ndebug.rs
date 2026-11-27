//@ run-pass
//@ compile-flags: -C debug_assertions=no

fn main() {
    let mut it = (u8::MAX - 2)..;
    assert_eq!(it.next().unwrap(), 253);
    assert_eq!(it.next().unwrap(), 254);
    assert_eq!(it.next(), None);

    let mut it = (i8::MAX - 2)..;
    assert_eq!(it.next().unwrap(), 125);
    assert_eq!(it.next().unwrap(), 126);
    assert_eq!(it.next(), None);
}
