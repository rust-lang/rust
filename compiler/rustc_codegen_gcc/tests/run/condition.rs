// Compiler:
//
// Run-time:
//   status: 0
//   stdout: true
//     1

#![feature(no_core, start)]

#![no_std]
#![no_core]

extern crate mini_core;

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        if argc == 1 {
            libc::printf(b"true\n\0" as *const u8 as *const i8);
        }

        let string =
            match argc {
                1 => b"1\n\0",
                2 => b"2\n\0",
                3 => b"3\n\0",
                4 => b"4\n\0",
                5 => b"5\n\0",
                _ => b"_\n\0",
            };
        libc::printf(string as *const u8 as *const i8);
    }
    0
}
