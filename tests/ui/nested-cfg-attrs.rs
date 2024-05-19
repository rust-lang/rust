#[cfg_attr(all(), cfg_attr(all(), cfg(FALSE)))]
fn f() {}

fn main() { f() } //~ ERROR cannot find function `f` in this scope
