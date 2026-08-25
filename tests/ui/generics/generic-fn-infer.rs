//@ run-pass




// Issue #45: infer type parameters in function applications

fn id<T>(x: T) -> T { return x; }

pub fn main() { let x: isize = 42; let y: isize = id(x); assert_eq!(x, y); }
