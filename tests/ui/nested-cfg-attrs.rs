#[cfg_attr(all(), cfg_attr(all(), cfg(FALSE)))] //~ NOTE the item is gated here
fn f() {} //~ NOTE found an item that was configured out

fn main() { f() } //~ ERROR cannot find function `f` in this scope
//~^ NOTE not found in this scope
