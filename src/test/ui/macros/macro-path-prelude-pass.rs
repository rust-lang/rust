// compile-pass

#![feature(extern_prelude)]

mod m {
    fn check() {
        std::panic!(); // OK
    }
}

fn main() {}
