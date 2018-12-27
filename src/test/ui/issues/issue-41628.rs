// compile-pass
#![deny(dead_code)]

#[used]
static FOO: u32 = 0;

fn main() {}
