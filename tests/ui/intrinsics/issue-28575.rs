#![feature(intrinsics)]

extern "C" {
    pub static FOO: extern "rust-intrinsic" fn();
}

fn main() {
    FOO() //~ ERROR: use of extern static is unsafe
}
