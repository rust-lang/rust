// build-pass (FIXME(62277): could be check-pass?)

#![feature(extern_prelude)]

mod m {
    fn check() {
        std::panic!(); // OK
    }
}

fn main() {}
