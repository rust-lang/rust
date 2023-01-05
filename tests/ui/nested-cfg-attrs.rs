#[cfg_attr(all(), cfg_attr(all(), cfg(foo)))]
fn f() {}

fn main() { f() } //~ ERROR cannot find function `f` in this scope
