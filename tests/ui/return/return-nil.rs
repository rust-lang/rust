//@ check-pass

fn f() { let x = (); return x; }

pub fn main() { f(); }
