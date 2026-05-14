#![deny(unused_unconstructable_pub_structs)]

pub struct ReceiverOnly(i32);
//~^ ERROR: struct `ReceiverOnly` is never constructed

impl ReceiverOnly {
    pub fn method(&self) {}
}

impl Clone for ReceiverOnly {
    fn clone(&self) -> ReceiverOnly {
        ReceiverOnly(0)
    }
}

pub trait Trait {
    fn method(&self);
}

impl Trait for ReceiverOnly {
    fn method(&self) {
        self.0;
    }
}

fn main() {}
