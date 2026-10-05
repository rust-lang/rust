//! Test that span parents (which are incremental only) are ignored when deduplicating diagnostics.
//! Regression test for #162901.

//@ revisions: dedup dup incr-dedup incr-dup
//@[dedup] compile-flags: -Z deduplicate-diagnostics=yes
//@[incr-dup] incremental
//@[incr-dedup] incremental
//@[incr-dedup] compile-flags: -Z deduplicate-diagnostics=yes

#[global_allocator]
static A: usize = 0;
//[dedup,dup,incr-dedup,incr-dup]~^ ERROR E0277
//[dup,incr-dup]~| ERROR E0277
//[dup,incr-dup]~| ERROR E0277
//[dup,incr-dup]~| ERROR E0277

fn main() {}
