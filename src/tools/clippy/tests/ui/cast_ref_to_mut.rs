#![warn(clippy::cast_ref_to_mut)]
#![allow(clippy::no_effect, clippy::borrow_as_ptr)]

extern "C" {
    // N.B., mutability can be easily incorrect in FFI calls -- as
    // in C, the default is mutable pointers.
    fn ffi(c: *mut u8);
    fn int_ffi(c: *mut i32);
}

fn main() {
    let s = String::from("Hello");
    let a = &s;
    unsafe {
        let num = &3i32;
        let mut_num = &mut 3i32;
        // Should be warned against
        (*(a as *const _ as *mut String)).push_str(" world");
        *(a as *const _ as *mut _) = String::from("Replaced");
        *(a as *const _ as *mut String) += " world";
        // Shouldn't be warned against
        println!("{}", *(num as *const _ as *const i16));
        println!("{}", *(mut_num as *mut _ as *mut i16));
        ffi(a.as_ptr() as *mut _);
        int_ffi(num as *const _ as *mut _);
        int_ffi(&3 as *const _ as *mut _);
        let mut value = 3;
        let value: *const i32 = &mut value;
        *(value as *const i16 as *mut i16) = 42;
    }
}
