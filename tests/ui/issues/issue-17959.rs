extern crate core;

use core::ops::Drop;

trait Bar {}

struct G<T: ?Sized> {
    _ptr: *const T
}

impl<T> Drop for G<T> {
//~^ ERROR `Drop` impl requires `T: Sized`
    fn drop(&mut self) {
        if !self._ptr.is_null() {
        }
    }
}

fn main() {
    let x:G<dyn Bar>;
}
