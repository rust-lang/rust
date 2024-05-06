// skip-filecheck
//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

use std::ops::{Range, RangeInclusive};

// EMIT_MIR range_iter.range_iter_next.PreCodegen.after.mir
pub fn range_iter_next(it: &mut Range<u32>) -> Option<u32> {
    it.next()
}

// EMIT_MIR range_iter.range_inclusive_iter_next.PreCodegen.after.mir
pub fn range_inclusive_iter_next(it: &mut RangeInclusive<u32>) -> Option<u32> {
    it.next()
}

// EMIT_MIR range_iter.forward_loop.PreCodegen.after.mir
pub fn forward_loop(start: u32, end: u32, f: impl Fn(u32)) {
    for x in start..end {
        f(x)
    }
}

// EMIT_MIR range_iter.inclusive_loop.PreCodegen.after.mir
pub fn inclusive_loop(start: u32, end: u32, f: impl Fn(u32)) {
    for x in start..=end {
        f(x)
    }
}
