pub fn main() { assert!((even(42))); assert!((odd(45))); }

fn even(n: isize) -> bool { if n == 0 { return true; } else { return odd(n - 1); } }

fn odd(n: isize) -> bool { if n == 0 { return false; } else { return even(n - 1); } }
