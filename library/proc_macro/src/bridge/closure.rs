//! Closure type (equivalent to `&mut dyn FnMut(A) -> R`) that's `repr(C)`.

use std::marker::PhantomData;

#[repr(C)]
pub struct Closure<'a, A, R> {
    call: unsafe extern "C" fn(*mut Env, A) -> R,
    env: *mut Env,
    // Ensure Closure is !Send and !Sync
    _marker: PhantomData<*mut &'a mut ()>,
}

struct Env;

impl<'a, A, R, F: FnMut(A) -> R> From<&'a mut F> for Closure<'a, A, R> {
    fn from(f: &'a mut F) -> Self {
        unsafe extern "C" fn call<A, R, F: FnMut(A) -> R>(env: *mut Env, arg: A) -> R {
            (*(env as *mut _ as *mut F))(arg)
        }
        Closure { call: call::<A, R, F>, env: f as *mut _ as *mut Env, _marker: PhantomData }
    }
}

impl<'a, A, R> Closure<'a, A, R> {
    pub fn call(&mut self, arg: A) -> R {
        unsafe { (self.call)(self.env, arg) }
    }
}
