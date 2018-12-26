#![feature(extern_prelude)]

mod m {
    fn check() {
        Vec::clone!(); //~ ERROR failed to resolve: not a module `Vec`
        u8::clone!(); //~ ERROR failed to resolve: not a module `u8`
    }
}

fn main() {}
