// pp-exact

fn from_native_fn(x: native fn()) { }
fn from_closure(x: fn()) { }
fn from_stack_closure(x: fn&()) { }
fn from_box_closure(x: fn@()) { }
fn from_unique_closure(x: fn~()) { }
fn main() { }
