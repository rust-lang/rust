//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::cell::UnsafeCell;

#[repr(C)]
#[derive(Default)]
struct Node {
    _meta: UnsafeCell<usize>,
    value: usize,
}

impl Node {
    fn value(&self) -> &usize {
        &self.value
    }
}

/// This used to cause Stacked Borrows errors because of trouble around conversion
/// from Box to raw pointer.
fn main() {
    unsafe {
        let a = Box::into_raw(Box::new(Node::default()));
        let ptr = &*a;
        *UnsafeCell::raw_get(a.cast::<UnsafeCell<usize>>()) = 2;
        assert_eq!(*ptr.value(), 0);
        drop(Box::from_raw(a));
    }
}
