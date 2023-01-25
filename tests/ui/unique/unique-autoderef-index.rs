// run-pass
// compile-flags: -C opt-level=0

#![allow(unused_must_use, unused_variables)]
#![feature(rustc_attrs)]

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
pub struct Unique<T: ?Sized> {
    pointer: *const T,
}

impl<T: ?Sized> Unique<T> {
    pub fn as_ptr(&self) -> *mut T {
        self.pointer as *const T as *mut T
    }
}

struct Droppy(Unique<i32>);

impl Drop for Droppy {
    fn drop(&mut self) {
        self.0.as_ptr();
    }
}

pub fn main() {
    let mut x = 42;
    Box::new(Droppy(unsafe { Unique { pointer: &mut x as *const i32 as _ } }));
}
