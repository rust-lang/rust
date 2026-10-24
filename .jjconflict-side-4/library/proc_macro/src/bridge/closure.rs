//! Closure type (equivalent to `&mut dyn FnMut(Buffer) -> Buffer`) that's `repr(C)`.

use std::marker::PhantomData;

use super::Buffer;

#[repr(C)]
pub(super) struct Closure<'a> {
    call: extern "C" fn(*mut Env, Buffer) -> Buffer,
    env: *mut Env,
    // Prevent Send and Sync impls.
    //
    // The `'a` lifetime parameter represents the lifetime of `Env`.
    _marker: PhantomData<*mut &'a mut ()>,
}

struct Env;

impl<'a, F: FnMut(Buffer) -> Buffer> From<&'a mut F> for Closure<'a> {
    fn from(f: &'a mut F) -> Self {
        extern "C" fn call<F: FnMut(Buffer) -> Buffer>(env: *mut Env, arg: Buffer) -> Buffer {
            unsafe { (*(env as *mut _ as *mut F))(arg) }
        }
        Closure { call: call::<F>, env: f as *mut _ as *mut Env, _marker: PhantomData }
    }
}

impl<'a> Closure<'a> {
    pub(super) fn call(&mut self, arg: Buffer) -> Buffer {
        (self.call)(self.env, arg)
    }
}
