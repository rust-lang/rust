fn f() -> isize { { return 3; } }

pub fn main() { assert_eq!(f(), 3); }
