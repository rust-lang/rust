// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

mod foo {
    extern {
        pub static errno: u32;
    }
}

pub fn main() {
}
