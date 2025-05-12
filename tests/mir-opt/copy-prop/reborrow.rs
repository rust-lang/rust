// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Check that CopyProp considers reborrows as not mutating the pointer.
//@ test-mir-pass: CopyProp

#[inline(never)]
fn opaque(_: impl Sized) {}

// EMIT_MIR reborrow.remut.CopyProp.diff
fn remut(mut x: u8) {
    let a = &mut x;
    let b = &mut *a; //< this cannot mutate a.
    let c = a; //< so `c` and `a` can be merged.
    opaque(c);
}

// EMIT_MIR reborrow.reraw.CopyProp.diff
fn reraw(mut x: u8) {
    let a = &mut x;
    let b = &raw mut *a; //< this cannot mutate a.
    let c = a; //< so `c` and `a` can be merged.
    opaque(c);
}

// EMIT_MIR reborrow.miraw.CopyProp.diff
fn miraw(mut x: u8) {
    let a = &raw mut x;
    let b = unsafe { &raw mut *a }; //< this cannot mutate a.
    let c = a; //< so `c` and `a` can be merged.
    opaque(c);
}

// EMIT_MIR reborrow.demiraw.CopyProp.diff
fn demiraw(mut x: u8) {
    let a = &raw mut x;
    let b = unsafe { &mut *a }; //< this cannot mutate a.
    let c = a; //< so `c` and `a` can be merged.
    opaque(c);
}

fn main() {
    remut(0);
    reraw(0);
    miraw(0);
    demiraw(0);
}
