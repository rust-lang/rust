// compile-flags: -O -Zmir-opt-level=2 -g
// ignore-debug

#![crate_type = "lib"]

pub fn int_range(start: usize, end: usize) {
    for i in start..end {
        opaque(i)
    }
}

pub fn vec_range(mut v: Vec<impl Sized>) {
    for i in 0..v.len() {
        let x = &mut v[i];
        opaque((i, x))
    }
    for i in 0..v.len() {
        let x = &v[i];
        opaque((i, x))
    }
}

pub fn vec_iter(mut v: Vec<impl Sized>) {
    for x in v.iter_mut() {
        opaque(x)
    }
    for x in v.iter() {
        opaque(x)
    }
}

pub fn vec_iter_enumerate(mut v: Vec<impl Sized>) {
    for (i, x) in v.iter_mut().enumerate() {
        opaque((i, x))
    }
    for (i, x) in v.iter().enumerate() {
        opaque((i, x))
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
// EMIT_MIR loops.vec_range.PreCodegen.after.mir
// EMIT_MIR loops.vec_iter.PreCodegen.after.mir
// EMIT_MIR loops.vec_iter_enumerate.PreCodegen.after.mir
// EMIT_MIR loops.vec_move.PreCodegen.after.mir
