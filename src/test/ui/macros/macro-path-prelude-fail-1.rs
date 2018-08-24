#![feature(extern_prelude)]

mod m {
    fn check() {
        Vec::clone!(); //~ ERROR failed to resolve. Not a module `Vec`
        u8::clone!(); //~ ERROR failed to resolve. Not a module `u8`
    }
}

fn main() {}
