//! Closure type (equivalent to `&mut dyn FnMut(A) -> R`) that's `repr(C)`.

use std::marker::PhantomData;

#[repr(C)]
pub(super) struct Closure<'a, A, R> {
    call: unsafe extern "C" fn(*mut Env, A) -> R,
    env: *mut Env,
    // Prevent Send and Sync impls. `!Send`/`!Sync` is the usual way of doing
    // this, but that requires unstable features. rust-analyzer uses this code
    // and avoids unstable features.
    //
    // The `'a` lifetime parameter represents the lifetime of `Env`.
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
    pub(super) fn call(&mut self, arg: A) -> R {
        unsafe { (self.call)(self.env, arg) }
    }
}
