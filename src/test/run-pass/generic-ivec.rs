// xfail-stage0

fn f[T](v: @T) { }
fn main() { f(@~[1, 2, 3, 4, 5]); }

