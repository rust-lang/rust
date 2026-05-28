//@ run-pass



pub fn main() { let x: isize = 10; while x == 10 && x == 11 { let _y = 0xf00_usize; } }
