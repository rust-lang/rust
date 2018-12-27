//! Closure type (equivalent to `&mut dyn FnMut(A) -> R`) that's `repr(C)`.

#[repr(C)]
pub struct Closure<'a, A, R> {
    call: unsafe extern "C" fn(&mut Env, A) -> R,
    env: &'a mut Env,
}

extern "C" {
    type Env;
}

impl<'a, A, R> !Sync for Closure<'a, A, R> {}
impl<'a, A, R> !Send for Closure<'a, A, R> {}

impl<'a, A, R, F: FnMut(A) -> R> From<&'a mut F> for Closure<'a, A, R> {
    fn from(f: &'a mut F) -> Self {
        unsafe extern "C" fn call<A, R, F: FnMut(A) -> R>(env: &mut Env, arg: A) -> R {
            (*(env as *mut _ as *mut F))(arg)
        }
        Closure {
            call: call::<A, R, F>,
            env: unsafe { &mut *(f as *mut _ as *mut Env) },
        }
    }
}

impl<'a, A, R> Closure<'a, A, R> {
    pub fn call(&mut self, arg: A) -> R {
        unsafe { (self.call)(self.env, arg) }
    }
}
