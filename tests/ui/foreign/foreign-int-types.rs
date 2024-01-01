// build-pass
#![forbid(improper_ctypes)]


mod xx {
    extern "C" {
        pub fn strlen(str: *const u8) -> usize;
        pub fn foo(x: isize, y: usize);
    }
}

fn main() {}
