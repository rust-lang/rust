#![deny(variant_size_differences)]

enum _En {
    V0(u8),
    VBig([u8; 1024]),   //~ ERROR variant is more than three times larger
}

fn main() {}
