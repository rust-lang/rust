use std::cell::UnsafeCell;
use std::mem;

pub struct VecCell<T> {
    data: UnsafeCell<Vec<T>>
}

impl<T> VecCell<T> {
    pub fn with_capacity(capacity: usize) -> VecCell<T>{
        VecCell { data: UnsafeCell::new(Vec::with_capacity(capacity)) }
    }

    #[inline]
    pub fn push(&self, data: T) -> usize {
        // The logic here, and in `swap` below, is that the `push`
        // method on the vector will not recursively access this
        // `VecCell`. Therefore, we can temporarily obtain mutable
        // access, secure in the knowledge that even if aliases exist
        // -- indeed, even if aliases are reachable from within the
        // vector -- they will not be used for the duration of this
        // particular fn call. (Note that we also are relying on the
        // fact that `VecCell` is not `Sync`.)
        unsafe {
            let v = self.data.get();
            (*v).push(data);
            (*v).len()
        }
    }

    pub fn swap(&self, mut data: Vec<T>) -> Vec<T> {
        unsafe {
            let v = self.data.get();
            mem::swap(&mut *v, &mut data);
        }
        data
    }
}
