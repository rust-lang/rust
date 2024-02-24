//@ revisions:rpass1 rpass2

#[cfg(rpass1)]
pub static A: i32 = 42;
#[cfg(rpass2)]
pub static A: i32 = 43;

pub static B: &i32 = &A;

fn main() {}
