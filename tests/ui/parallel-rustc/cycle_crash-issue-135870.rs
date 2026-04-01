// Test for #135870, which causes a deadlock bug
//
//@ compile-flags: -Z threads=2
//@ compare-output-by-lines

const FOO: usize = FOO; //~ ERROR cycle detected

fn main() {}
