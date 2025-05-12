//@ run-pass
pub struct Z(#[allow(dead_code)] &'static Z);

pub fn main() {}
