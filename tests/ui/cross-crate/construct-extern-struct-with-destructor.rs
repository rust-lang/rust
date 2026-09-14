//@ run-pass
//@ aux-build:construct-extern-struct-with-destructor.rs

//! Regression test for https://github.com/rust-lang/rust/issues/3012
//! Guarantees that you can construct cross crate structs.

extern crate construct_extern_struct_with_destructor as socketlib;

use socketlib::socket;

pub fn main() {
    let fd: u32 = 1 as u32;
    let _sock: Box<_> = Box::new(socket::socket_handle(fd));
}
