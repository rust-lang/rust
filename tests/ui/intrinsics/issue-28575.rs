#![feature(intrinsics)]

extern "C" {
    pub static FOO: extern "rust-intrinsic" fn();
    //~^ ERROR invalid ABI
}

fn main() {
    FOO() //~ ERROR: use of extern static is unsafe
}
