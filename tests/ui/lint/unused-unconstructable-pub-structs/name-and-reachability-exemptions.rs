//@ check-pass

#![deny(unused_unconstructable_pub_structs)]

pub struct _Prefixed(i32);

mod private {
    pub struct Unreachable(i32);
}

fn main() {}
