// Check that `Allocator` is dyn-incompatible, to keep the design space open.

#![feature(allocator_ext)]

use std::alloc::{Allocator, System};

fn ensure_dyn_incompatible(_: &dyn Allocator) {}
//~^ ERROR E0038

fn main() {
    ensure_dyn_incompatible(&System);
    //~^ ERROR E0038
}
