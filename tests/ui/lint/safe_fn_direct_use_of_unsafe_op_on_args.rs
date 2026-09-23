//@ build-pass

pub fn myfunc_ptr_unary(ptr: *mut u32) {
    unsafe { let _ = *ptr; }
    //~^ WARN call to an unsafe function on an argument of a safe function
}

pub fn myfunc_ptr_methodcall(ptr: *mut u32) {
    unsafe { ptr.write(123); }
    //~^ WARN call to an unsafe function on an argument of a safe function
}

pub fn myfunc_ptr_call(ptr: *mut u32) {
    unsafe { std::ptr::write(ptr, 123); }
    //~^ WARN call to an unsafe function on an argument of a safe function
}

pub fn myfunc_u32_methodcall(i: u32) -> u32 {
    let j;

    unsafe { j = i.unchecked_add(1); }
    //~^ WARN call to an unsafe function on an argument of a safe function
    j
}

fn main() {
    myfunc_ptr_unary(0 as _);
    myfunc_ptr_methodcall(0 as _);
    myfunc_ptr_call(0 as _);
    myfunc_u32_methodcall(42);
}
