// build-pass

// pretty-expanded FIXME #23616

mod foo {
    extern "C" {
        pub static errno: u32;
    }
}

pub fn main() {}
