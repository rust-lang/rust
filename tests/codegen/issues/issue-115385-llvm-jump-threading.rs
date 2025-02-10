//@ compile-flags: -Copt-level=3 -Ccodegen-units=1

#![crate_type = "lib"]

#[repr(i64)]
pub enum Boolean {
    False = 0,
    True = 1,
}

impl Clone for Boolean {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Boolean {}

extern "C" {
    fn set_value(foo: *mut i64);
    fn bar();
}

pub fn foo(x: bool) {
    let mut foo = core::mem::MaybeUninit::<i64>::uninit();
    unsafe {
        set_value(foo.as_mut_ptr());
    }

    if x {
        let l1 = unsafe { *foo.as_mut_ptr().cast::<Boolean>() };
        if matches!(l1, Boolean::False) {
            unsafe {
                *foo.as_mut_ptr() = 0;
            }
        }
    }

    let l2 = unsafe { *foo.as_mut_ptr() };
    if l2 == 2 {
        // CHECK: call void @bar
        unsafe {
            bar();
        }
    }
}
