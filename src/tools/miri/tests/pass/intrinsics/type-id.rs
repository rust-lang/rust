use std::any::{Any, TypeId};

fn main() {
    let t1 = TypeId::of::<u64>();
    let t2 = TypeId::of::<u64>();
    assert_eq!(t1, t2);
    let t3 = TypeId::of::<usize>();
    assert_ne!(t1, t3);

    let _ = format!("{t1:?}"); // test that we can debug-print

    let b = Box::new(0u64) as Box<dyn Any>;
    assert_eq!(*b.downcast_ref::<u64>().unwrap(), 0);
    assert!(b.downcast_ref::<usize>().is_none());

    // Get the first pointer chunk and try to make it a ZST ref.
    // This used to trigger an error because TypeId allocs got misclassified as "LiveData".
    let _raw_chunk = unsafe { (&raw const t1).cast::<&()>().read() };
}
