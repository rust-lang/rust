fn make_box() -> Box<(i16, i16)> {
    Box::new((1, 2))
}

fn allocate_reallocate() {
    let mut s = String::new();

    // 6 byte heap alloc (__rust_allocate)
    s.push_str("foobar");
    assert_eq!(s.len(), 6);
    assert_eq!(s.capacity(), 8);

    // heap size doubled to 12 (__rust_reallocate)
    s.push_str("baz");
    assert_eq!(s.len(), 9);
    assert_eq!(s.capacity(), 16);

    // heap size reduced to 9  (__rust_reallocate)
    s.shrink_to_fit();
    assert_eq!(s.len(), 9);
    assert_eq!(s.capacity(), 9);
}

fn main() {
    assert_eq!(*make_box(), (1, 2));
    allocate_reallocate();
}
