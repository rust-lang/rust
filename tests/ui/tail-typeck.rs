//@error-in-other-file: mismatched types

fn f() -> isize { return g(); }

fn g() -> usize { return 0; }

fn main() { let y = f(); }
