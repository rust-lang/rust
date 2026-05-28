// Documents that Rust currently does not permit the coercion &mut &mut T -> *mut *mut T
// Making this compile was a feature request in rust-lang/rust#34117 but this is currently
// "working as intended". Allowing "deep pointer coercion" seems footgun-prone, and would
// require proceeding carefully.

//@ dont-require-annotations: NOTE

use std::ops::{Deref, DerefMut};

struct Foo(i32);

struct SmartPtr<T>(*mut T);

impl<T> SmartPtr<T> {
    fn get_addr(&mut self) -> &mut *mut T {
        &mut self.0
    }
}

impl<T> Deref for SmartPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.0 }
    }
}
impl<T> DerefMut for SmartPtr<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}

/// Puts a Foo into the pointer provided by the caller
fn make_foo(_: *mut *mut Foo) {
    unimplemented!()
}

fn main() {
    let mut result: SmartPtr<Foo> = SmartPtr(std::ptr::null_mut());
    make_foo(&mut &mut *result); //~ ERROR mismatched types
                                 //~^ NOTE expected `*mut *mut Foo`, found `&mut &mut Foo`
    make_foo(out(&mut result)); // works, but makes one wonder why above coercion cannot happen
}

fn out<T>(ptr: &mut SmartPtr<T>) -> &mut *mut T {
    ptr.get_addr()
}
