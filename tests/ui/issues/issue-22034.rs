fn main() {
    let ptr: *mut () = core::ptr::null_mut();
    let _: &mut dyn Fn() = unsafe {
        &mut *(ptr as *mut dyn Fn())
        //~^ ERROR cannot cast thin pointer `*mut ()` to wide pointer `*mut dyn Fn()`
    };
}
