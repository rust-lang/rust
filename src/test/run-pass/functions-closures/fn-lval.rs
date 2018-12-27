// run-pass



// pretty-expanded FIXME #23616

fn foo(_f: fn(isize) -> isize) { }

fn id(x: isize) -> isize { return x; }

pub fn main() { foo(id); }
