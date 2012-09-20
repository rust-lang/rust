fn f(a: *int) -> *int { return a; }

fn g(a: *int) -> *int { let b = f(a); return b; }

fn main() { return; }
