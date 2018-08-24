fn foo(x: &mut Vec<&'_ u8>, y: &'_ u8) { x.push(y); } //~ ERROR lifetime mismatch

fn main() {}
