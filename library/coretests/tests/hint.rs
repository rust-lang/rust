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

#[test]
#[should_panic = "message canary"]
fn select_unpredictable_drop_on_panic() {
    use core::cell::Cell;

    struct X<'a> {
        cell: &'a Cell<u16>,
        expect: u16,
        write: u16,
    }

    impl Drop for X<'_> {
        fn drop(&mut self) {
            let value = self.cell.get();
            self.cell.set(self.write);
            assert_eq!(value, self.expect, "message canary");
        }
    }

    let cell = Cell::new(0);

    // Trigger a double-panic if the selected cell was not dropped during panic.
    let _armed = X { cell: &cell, expect: 0xdead, write: 0 };
    let selected = X { cell: &cell, write: 0xdead, expect: 1 };
    let unselected = X { cell: &cell, write: 1, expect: 0xff };

    // The correct drop order is:
    //
    // 1. `unselected` drops, writes 1, and panics as 0 != 0xff
    // 2. `selected` drops during unwind, writes 0xdead and does not panic as 1 == 1
    // 3. `armed` drops during unwind, writes 0 and does not panic as 0xdead == 0xdead
    //
    // If `selected` is not dropped, `armed` panics as 1 != 0xdead
    let _unreachable = core::hint::select_unpredictable(true, selected, unselected);
}
