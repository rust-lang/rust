#![sanitizer_runtime]
#![feature(alloc_system)]
#![cfg_attr(not(stage0), feature(nll))]
#![feature(sanitizer_runtime)]
#![feature(staged_api)]
#![no_std]
#![unstable(feature = "sanitizer_runtime_lib",
            reason = "internal implementation detail of sanitizers",
            issue = "0")]

extern crate alloc_system;

use alloc_system::System;

#[global_allocator]
static ALLOC: System = System;
