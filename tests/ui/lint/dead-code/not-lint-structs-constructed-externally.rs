//@ check-pass
#![deny(dead_code)]

#[repr(C)]
pub struct Foo {
    pub i: i16,
    align: i16
}

mod ffi {
    use super::*;

    extern "C" {
        pub fn DomPromise_AddRef(promise: *const Promise);
        pub fn DomPromise_Release(promise: *const Promise);
    }
}

#[repr(C)]
pub struct Promise {
    private: [u8; 0],
    __nosync: ::std::marker::PhantomData<::std::rc::Rc<u8>>,
}

pub unsafe trait RefCounted {
    unsafe fn addref(&self);
    unsafe fn release(&self);
}

unsafe impl RefCounted for Promise {
    unsafe fn addref(&self) {
        ffi::DomPromise_AddRef(self)
    }
    unsafe fn release(&self) {
        ffi::DomPromise_Release(self)
    }
}

fn main() {}
