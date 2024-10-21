//@ known-bug: #130687
//@ only-x86_64
pub struct Data([u8; usize::MAX >> 16]);
const _: &'static Data = &Data([0; usize::MAX >> 16]);
