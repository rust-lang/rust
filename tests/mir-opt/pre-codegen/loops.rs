// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -g
//@ ignore-std-debug-assertions (debug assertions result in different inlines)
//@ needs-unwind

#![crate_type = "lib"]

pub fn int_range(start: usize, end: usize) {
    for i in start..end {
        opaque(i)
    }
}

pub fn mapped<T, U>(iter: impl Iterator<Item = T>, f: impl Fn(T) -> U) {
    for x in iter.map(f) {
        opaque(x)
    }
}

pub fn filter_mapped<T, U>(iter: impl Iterator<Item = T>, f: impl Fn(T) -> Option<U>) {
    for x in iter.filter_map(f) {
        opaque(x)
    }
}

pub fn vec_move(mut v: Vec<impl Sized>) {
    for x in v {
        opaque(x)
    }
}

#[inline(never)]
fn opaque(_: impl Sized) {}

// EMIT_MIR loops.int_range.PreCodegen.after.mir
// EMIT_MIR loops.mapped.PreCodegen.after.mir
// EMIT_MIR loops.filter_mapped.PreCodegen.after.mir
// EMIT_MIR loops.vec_move.PreCodegen.after.mir
