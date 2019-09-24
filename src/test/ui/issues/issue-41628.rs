// build-pass (FIXME(62277): could be check-pass?)
#![deny(dead_code)]

#[used]
static FOO: u32 = 0;

fn main() {}
