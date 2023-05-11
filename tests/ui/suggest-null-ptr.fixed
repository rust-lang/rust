// run-rustfix

// Suggest providing a std::ptr::null{,_mut}() to a function that takes in a raw
// pointer if a literal 0 was provided by the user.

extern "C" {
    fn foo(ptr: *const u8);

    fn foo_mut(ptr: *mut u8);

    fn usize(ptr: *const usize);

    fn usize_mut(ptr: *mut usize);
}

fn main() {
    unsafe {
        foo(std::ptr::null());
        //~^ mismatched types [E0308]
        //~| if you meant to create a null pointer, use `std::ptr::null()`
        foo_mut(std::ptr::null_mut());
        //~^ mismatched types [E0308]
        //~| if you meant to create a null pointer, use `std::ptr::null_mut()`
        usize(std::ptr::null());
        //~^ mismatched types [E0308]
        //~| if you meant to create a null pointer, use `std::ptr::null()`
        usize_mut(std::ptr::null_mut());
        //~^ mismatched types [E0308]
        //~| if you meant to create a null pointer, use `std::ptr::null_mut()`
    }
}
