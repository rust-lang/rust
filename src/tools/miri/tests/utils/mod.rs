#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
mod macros;

mod fs;
mod miri_extern;

pub use fs::*;
pub use miri_extern::*;

pub fn run_provenance_gc() {
    // SAFETY: No preconditions. The GC is fine to run at any time.
    unsafe {
        miri_run_provenance_gc()
    }
}
