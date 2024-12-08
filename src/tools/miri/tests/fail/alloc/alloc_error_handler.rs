//@error-in-other-file: aborted
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
#![feature(allocator_api)]

use std::alloc::*;

fn main() {
    handle_alloc_error(Layout::for_value(&0));
}
