extern "C" {
    fn printf(c: *const i8, ...);
}

fn main() {
    unsafe {
        printf(::std::ptr::null(), 0f32);
        //~^ ERROR can't pass `f32` to variadic function
        //~| HELP cast the value to `c_double`
        //~| HELP cast the value to `c_double`
        printf(::std::ptr::null(), 0i8);
        //~^ ERROR can't pass `i8` to variadic function
        //~| HELP cast the value to `c_int`
        //~| HELP cast the value to `c_int`
        printf(::std::ptr::null(), 0i16);
        //~^ ERROR can't pass `i16` to variadic function
        //~| HELP cast the value to `c_int`
        //~| HELP cast the value to `c_int`
        printf(::std::ptr::null(), 0u8);
        //~^ ERROR can't pass `u8` to variadic function
        //~| HELP cast the value to `c_uint`
        //~| HELP cast the value to `c_uint`
        printf(::std::ptr::null(), 0u16);
        //~^ ERROR can't pass `u16` to variadic function
        //~| HELP cast the value to `c_uint`
        //~| HELP cast the value to `c_uint`
        printf(::std::ptr::null(), printf);
        //~^ ERROR can't pass `{fn item printf: unsafe extern "C" fn(*const i8, ...)}` to variadic function
        //~| HELP a function item is zero-sized and needs to be casted into a function pointer to be used in FFI
        //~| HELP cast the value into a function pointer
        //~| HELP cast the value to `unsafe extern "C" fn(*const i8, ...)`
    }
}
