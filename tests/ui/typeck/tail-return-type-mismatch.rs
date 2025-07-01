fn f() -> isize { return g(); } //~ ERROR mismatched types

fn g() -> usize { return 0; }

fn main() { let y = f(); }
