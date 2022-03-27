// stderr-per-bitwidth
use std::mem;

static FOO: bool = unsafe { mem::transmute(3u8) };
//~^ ERROR could not evaluate static initializer

fn main() {}
