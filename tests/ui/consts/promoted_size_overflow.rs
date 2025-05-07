//@ only-64bit
pub struct Data([u8; usize::MAX >> 2]);
const _: &'static [Data] = &[];
//~^ERROR: evaluation of constant value failed
//~| NOTE too big for the target architecture

fn main() {}
