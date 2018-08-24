// pp-exact

fn from_foreign_fn(_x: fn()) { }
fn from_stack_closure<F>(_x: F) where F: Fn() { }
fn main() { }
