//@ check-pass

pub static STATIC: u32 = 0;
pub struct Foo<const N: u32>;
pub const FOO: Foo<{STATIC}> = Foo;

fn main() {}
