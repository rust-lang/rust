//@ revisions: bfail1 bfail2
//@ build-pass
//@ needs-crate-type: cdylib

#![crate_type="lib"]
#![crate_type="cdylib"]

#[allow(unused_imports)]
use std::alloc::System;

#[cfg(bfail1)]
#[global_allocator]
static ALLOC: System = System;
