fn f(x: Box<[u8; 500]>) {}
//~^ ERROR: local variable doesn't need to be boxed here
fn f2(x: Box<[u8; 501]>) {}

fn main() {}
