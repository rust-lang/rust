#![feature(box_syntax)]

fn make_box() -> Box<(i16, i16)> {
    Box::new((1, 2))
}

fn make_box_syntax() -> Box<(i16, i16)> {
    box (1, 2)
}

fn allocate_reallocate() {
    let mut s = String::new();

    // 4 byte heap alloc (__rust_allocate)
    s.push('f');
    assert_eq!(s.len(), 1);
    assert_eq!(s.capacity(), 4);

    // heap size doubled to 8 (__rust_reallocate)
    // FIXME: String::push_str is broken because it hits the std::vec::SetLenOnDrop code and we
    // don't call destructors in miri yet.
    s.push('o');
    s.push('o');
    s.push('o');
    s.push('o');
    assert_eq!(s.len(), 5);
    assert_eq!(s.capacity(), 8);

    // heap size reduced to 5 (__rust_reallocate)
    s.shrink_to_fit();
    assert_eq!(s.len(), 5);
    assert_eq!(s.capacity(), 5);
}

fn main() {
    assert_eq!(*make_box(), (1, 2));
    assert_eq!(*make_box_syntax(), (1, 2));
    allocate_reallocate();
}
