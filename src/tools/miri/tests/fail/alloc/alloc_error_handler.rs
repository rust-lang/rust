//@error-in-other-file: aborted
#![feature(allocator_api)]

use std::alloc::*;

fn main() {
    handle_alloc_error(Layout::for_value(&0));
}
