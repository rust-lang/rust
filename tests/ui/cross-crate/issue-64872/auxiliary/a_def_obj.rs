//@ compile-flags: -C debuginfo=2

//@ no-prefer-dynamic
#![crate_type = "rlib"]

pub trait Object { fn method(&self) { } }

impl Object for u32 { }
impl Object for () { }
impl<T> Object for &T { }

pub fn unused() {
    let ref u = 0_u32;
    let _d = &u as &dyn crate::Object;
    _d.method()
}
