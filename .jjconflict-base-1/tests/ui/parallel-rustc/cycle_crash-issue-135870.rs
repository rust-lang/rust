// Test for #135870, which causes a deadlock bug

const FOO: usize = FOO; //~ ERROR cycle detected

fn main() {}
