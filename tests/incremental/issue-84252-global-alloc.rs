//@ revisions: cfail1 cfail2
//@ build-pass

#![crate_type="lib"]
#![crate_type="cdylib"]

#[allow(unused_imports)]
use std::alloc::System;

#[cfg(cfail1)]
#[global_allocator]
static ALLOC: System = System;
