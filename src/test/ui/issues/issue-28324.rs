#![allow(safe_extern_statics)]

extern {
    static error_message_count: u32;
}

pub static BAZ: u32 = *&error_message_count;
//~^ ERROR could not evaluate static initializer
//~| tried to read from foreign (extern) static

fn main() {}
