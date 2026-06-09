//@ build-pass

#[cfg_attr(true, unsafe(no_mangle))]
fn a() {}

fn main() {}
