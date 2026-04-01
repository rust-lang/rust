fn main() {
    let ptr: *mut () = core::ptr::null_mut();
    let _: &mut dyn Fn() = unsafe {
        &mut *(ptr as *mut dyn Fn())
        //~^ ERROR expected a `Fn()` closure, found `()`
    };
}
