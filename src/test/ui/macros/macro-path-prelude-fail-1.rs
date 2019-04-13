#![feature(extern_prelude)]

mod m {
    fn check() {
        Vec::clone!(); //~ ERROR failed to resolve: `Vec` is a struct, not a module
        u8::clone!(); //~ ERROR failed to resolve: `u8` is a builtin type, not a module
    }
}

fn main() {}
