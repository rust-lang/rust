// run-rustfix

pub fn foo(x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

pub fn foo2(x: &mut Vec<&'_ u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

pub fn foo3<'a>(_other: &'a [u8], x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime mismatch

fn main() {}
