//@ run-pass




pub fn main() { let mut x: i32 = -400; x = 0 - x; assert_eq!(x, 400); }
