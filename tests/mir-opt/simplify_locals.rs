// skip-filecheck
//@ test-mir-pass: SimplifyLocals-before-const-prop

#![feature(thread_local)]

#[derive(Copy, Clone)]
enum E {
    A,
    B,
}

// EMIT_MIR simplify_locals.c.SimplifyLocals-before-const-prop.diff
fn c() {
    let bytes = [0u8; 10];
    // Unused cast
    let _: &[u8] = &bytes;
}

// EMIT_MIR simplify_locals.d1.SimplifyLocals-before-const-prop.diff
fn d1() {
    // Unused set discriminant
    let _ = E::A;
}

// EMIT_MIR simplify_locals.d2.SimplifyLocals-before-const-prop.diff
fn d2() {
    // Unused set discriminant
    { (10, E::A) }.1 = E::B;
}

// EMIT_MIR simplify_locals.r.SimplifyLocals-before-const-prop.diff
fn r() {
    let mut a = 1;
    // Unused references
    let _ = &a;
    let _ = &mut a;
}

#[thread_local]
static mut X: u32 = 0;

// EMIT_MIR simplify_locals.t1.SimplifyLocals-before-const-prop.diff
fn t1() {
    // Unused thread local
    unsafe { X };
}

// EMIT_MIR simplify_locals.t2.SimplifyLocals-before-const-prop.diff
fn t2() {
    // Unused thread local
    unsafe { &mut X };
}

// EMIT_MIR simplify_locals.t3.SimplifyLocals-before-const-prop.diff
fn t3() {
    // Unused thread local
    unsafe { *&mut X };
}

// EMIT_MIR simplify_locals.t4.SimplifyLocals-before-const-prop.diff
fn t4() -> u32 {
    // Used thread local
    unsafe { X + 1 }
}

// EMIT_MIR simplify_locals.expose_provenance.SimplifyLocals-before-const-prop.diff
fn expose_provenance(p: *const usize) {
    // Used pointer to address cast. Has a side effect of exposing the provenance.
    p as usize;
}

fn main() {
    c();
    d1();
    d2();
    r();
    t1();
    t2();
    t3();
    t4();
    expose_provenance(&0);
}
