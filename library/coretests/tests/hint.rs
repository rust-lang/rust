#[test]
fn select_unpredictable_drop() {
    use core::cell::Cell;
    struct X<'a>(&'a Cell<bool>);
    impl Drop for X<'_> {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    let a_dropped = Cell::new(false);
    let b_dropped = Cell::new(false);
    let a = X(&a_dropped);
    let b = X(&b_dropped);
    assert!(!a_dropped.get());
    assert!(!b_dropped.get());
    let selected = core::hint::select_unpredictable(core::hint::black_box(true), a, b);
    assert!(!a_dropped.get());
    assert!(b_dropped.get());
    drop(selected);
    assert!(a_dropped.get());
    assert!(b_dropped.get());
}
