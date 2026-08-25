#![feature(extern_types)]
#![feature(raw_dylib_elf)]

use std::ffi::c_char;

#[link(name = "extern", kind = "raw-dylib")]
unsafe extern "C" {
    type FOO;
    fn create_foo() -> *const FOO;
    fn get_foo(foo: *const FOO) -> c_char;
    fn set_foo(foo: *const FOO, value: c_char);
}

pub fn main() {
    let value = unsafe {
        let foo = create_foo();
        set_foo(foo, 42);
        get_foo(foo)
    };
    println!("{}", value);
}
