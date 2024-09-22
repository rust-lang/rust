//@ known-bug: rust-lang/rust#125476
//@ only-x86_64
pub struct Data([u8; usize::MAX >> 2]);
const _: &'static [Data] = &[];
