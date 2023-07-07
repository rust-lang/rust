// compile-flags: -Ztrait-solver=next
// check-pass

use std::any::Any;

fn needs_usize(_: &usize) {}

fn main() {
    let x: &dyn Any = &1usize;
    if let Some(x) = x.downcast_ref::<usize>() {
        needs_usize(x);
    }
}
