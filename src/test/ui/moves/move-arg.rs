// run-pass

fn test(foo: isize) { assert_eq!(foo, 10); }

pub fn main() { let x = 10; test(x); }
