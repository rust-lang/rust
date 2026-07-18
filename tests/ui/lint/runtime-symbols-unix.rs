// This test checks the runtime symbols lint with the Unix symbols.

//@ only-unix
//@ edition: 2021
//@ normalize-stderr: "\*const [iu]8" -> "*const U8"

#![feature(c_variadic)]
#![allow(clashing_extern_declarations)] // we are voluntarily testing different definitions

use core::ffi::{c_char, c_int, c_void};

fn invalid() {
    #[no_mangle]
    pub fn open() {}
    //~^ ERROR invalid definition of the runtime `open` symbol

    extern "C" {
        pub fn read();
        //~^ ERROR invalid definition of the runtime `read` symbol

        pub fn write();
        //~^ ERROR invalid definition of the runtime `write` symbol
    }

    #[no_mangle]
    pub static close: () = ();
    //~^ ERROR invalid definition of the runtime `close` symbol

    extern "C" {
        pub fn malloc();
        //~^ ERROR invalid definition of the runtime `malloc` symbol

        pub fn realloc();
        //~^ ERROR invalid definition of the runtime `realloc` symbol

        pub fn free();
        //~^ ERROR invalid definition of the runtime `free` symbol

        pub fn exit();
        //~^ ERROR invalid definition of the runtime `exit` symbol
    }
}

fn suspicious() {
    extern "C" {
        pub fn open(path: *const u8, oflag: usize, ...) -> c_int;
        //~^ WARN suspicious definition of the runtime `open` symbol

        pub fn free(ptr: *const u8);
        //~^ WARN suspicious definition of the runtime `free` symbol

        pub fn exit(code: f32) -> !;
        //~^ WARN suspicious definition of the runtime `exit` symbol

        #[link_name = "exit"]
        pub static exit2: Option<unsafe extern "C" fn(f32) -> !>;
        //~^ WARN suspicious definition of the runtime `exit` symbol

        #[link_name = "exit"]
        pub static exit3: unsafe extern "C" fn(f32) -> !;
        //~^ WARN suspicious definition of the runtime `exit` symbol
    }
}

fn main() {}
