// test the deref_nullptr lint

#![deny(deref_nullptr)]

fn f() {
    unsafe {
        let a = 1;
        let ub = *(a as *const i32);
        let ub = *(0 as *const i32);
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *core::ptr::null::<i32>();
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *core::ptr::null_mut::<i32>();
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *(core::ptr::null::<i16>() as *const i32);
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *(core::ptr::null::<i16>() as *mut i32 as *mut usize as *const u8);
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = &*core::ptr::null::<i32>();
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        core::ptr::addr_of!(*core::ptr::null::<i32>());
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        std::ptr::addr_of_mut!(*core::ptr::null_mut::<i32>());
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *std::ptr::null::<i32>();
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
        let ub = *std::ptr::null_mut::<i32>();
        //~^ ERROR Dereferencing a null pointer causes undefined behavior
    }
}

fn main() {}
