//@ run-pass
// Check that trait objects without a principal codegen properly.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem;

// Array is to make sure the size is not exactly pointer-size, so
// we can be sure we are measuring the right size in the
// `size_of_val` test.
struct SetOnDrop<'a>(&'a AtomicUsize, #[allow(dead_code)] [u8; 64]);
impl<'a> Drop for SetOnDrop<'a> {
    fn drop(&mut self) {
        self.0.store(self.0.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
}

trait TypeEq<V: ?Sized> {}
impl<T: ?Sized> TypeEq<T> for T {}
fn assert_types_eq<U: ?Sized, V: ?Sized>() where U: TypeEq<V> {}

fn main() {
    // Check that different ways of writing the same type are equal.
    assert_types_eq::<dyn Sync, dyn Sync + Sync>();
    assert_types_eq::<dyn Sync + Send, dyn Send + Sync>();
    assert_types_eq::<dyn Sync + Send + Sync, dyn Send + Sync>();

    // Check that codegen works.
    //
    // Using `AtomicUsize` here because `Cell<u32>` is not `Sync`, and
    // so can't be made into a `Box<dyn Sync>`.
    let c = AtomicUsize::new(0);
    {
        let d: Box<dyn Sync> = Box::new(SetOnDrop(&c, [0; 64]));

        assert_eq!(mem::size_of_val(&*d),
                   mem::size_of::<SetOnDrop>());
        assert_eq!(mem::align_of_val(&*d),
                   mem::align_of::<SetOnDrop>());
        assert_eq!(c.load(Ordering::Relaxed), 0);
    }
    assert_eq!(c.load(Ordering::Relaxed), 1);
}
