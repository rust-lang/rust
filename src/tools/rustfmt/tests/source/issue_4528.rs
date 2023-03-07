#![allow(clippy::no_effect)]
 
extern "C" {
 // N.B., mutability can be easily incorrect in FFI calls -- as
     // in C, the default is mutable pointers.
    fn ffi(c: *mut u8);
     fn int_ffi(c: *mut i32);
}