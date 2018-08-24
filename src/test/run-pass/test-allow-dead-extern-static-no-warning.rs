// compile-flags: --test

#![deny(dead_code)]

extern "C" {
    #[allow(dead_code)]
    static Qt: u64;
}

fn main() {}
