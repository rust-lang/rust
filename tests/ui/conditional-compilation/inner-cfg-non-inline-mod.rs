//@ check-pass

mod module_with_cfg;

mod module_with_cfg {} // Ok, the module above is configured away by an inner attribute.

fn main() {}
