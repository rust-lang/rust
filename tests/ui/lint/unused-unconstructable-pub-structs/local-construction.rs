//@ check-pass

#![deny(unused_unconstructable_pub_structs)]

pub struct Constructed(i32);

impl Constructed {
    pub fn construct_self() -> Self {
        Constructed(0)
    }
}

impl Clone for Constructed {
    fn clone(&self) -> Constructed {
        Constructed(0)
    }
}

pub trait Trait {
    fn method(&self);
}

impl Trait for Constructed {
    fn method(&self) {
        self.0;
    }
}

fn main() {}
