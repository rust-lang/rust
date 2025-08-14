//@ compile-flags: -Z threads=2

const FOO: usize = FOO; //~ERROR cycle detected when const-evaluating + checking `FOO`

fn main() {}
