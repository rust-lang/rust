extern crate core;

use core::ops::Drop;

trait Bar {}

struct G<T: ?Sized> {
    _ptr: *const T
}

impl<T> Drop for G<T> {
//~^ ERROR: The requirement `T: std::marker::Sized` is added only by the Drop impl. [E0367]
    fn drop(&mut self) {
        if !self._ptr.is_null() {
        }
    }
}

fn main() {
    let x:G<dyn Bar>;
}
