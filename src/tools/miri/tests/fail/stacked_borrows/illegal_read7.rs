// Creating a shared reference does not leak the data to raw pointers,
// not even when interior mutability is involved.

use std::cell::Cell;
use std::ptr;

fn main() {
    unsafe {
        let x = &mut Cell::new(0);
        let raw = x as *mut Cell<i32>;
        let x = &mut *raw;
        let _shr = &*x;
        // The state here is interesting because the top of the stack is [Unique, SharedReadWrite],
        // just like if we had done `x as *mut _`.
        // If we said that reading from a lower item is fine if the top item is `SharedReadWrite`
        // (one way to maybe preserve a stack discipline), then we could now read from `raw`
        // without invalidating `x`.  That would be bad!  It would mean that creating `shr`
        // leaked `x` to `raw`.
        let _val = ptr::read(raw);
        let _val = *x.get_mut(); //~ ERROR: /retag .* tag does not exist in the borrow stack/
    }
}
