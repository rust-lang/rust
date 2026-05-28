//! Regression test for https://github.com/rust-lang/rust/issues/47703
//@ check-pass

struct AtomicRefMut<'a> {
    value: &'a mut i32,
    borrow: AtomicBorrowRefMut,
}

struct AtomicBorrowRefMut {
}

impl Drop for AtomicBorrowRefMut {
    fn drop(&mut self) {
    }
}

fn map(orig: AtomicRefMut) -> AtomicRefMut {
    AtomicRefMut {
        value: orig.value,
        borrow: orig.borrow,
    }
}

fn main() {}
