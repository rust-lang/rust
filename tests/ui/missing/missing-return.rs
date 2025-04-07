//@ error-pattern: return

fn f() -> isize { } //~ ERROR mismatched types

fn main() { f(); }
