//@ revisions: bpass1 bpass2
//@ needs-crate-type: cdylib

#![crate_type="lib"]
#![crate_type="cdylib"]

#[allow(unused_imports)]
use std::alloc::System;

#[cfg(bpass1)]
#[global_allocator]
static ALLOC: System = System;
