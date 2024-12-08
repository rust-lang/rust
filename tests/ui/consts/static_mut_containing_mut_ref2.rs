#![allow(static_mut_refs)]

static mut STDERR_BUFFER_SPACE: u8 = 0;

pub static mut STDERR_BUFFER: () = unsafe {
    *(&mut STDERR_BUFFER_SPACE) = 42;
    //~^ ERROR could not evaluate static initializer
};

fn main() {}
