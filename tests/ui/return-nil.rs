// run-pass
// pretty-expanded FIXME #23616

fn f() { let x = (); return x; }

pub fn main() { f(); }
