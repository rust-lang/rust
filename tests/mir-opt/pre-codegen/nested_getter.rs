// compile-flags: -O -Cdebuginfo=0 -Zmir-opt-level=2
// only-64bit
// ignore-debug

#![crate_type = "lib"]

pub struct Outer {
    inner: Inner,
}

struct Inner {
    inner: u8
}

#[inline]
fn inner_get(this: &Inner) -> u8 {
    this.inner
}

// EMIT_MIR nested_getter.outer_get.PreCodegen.after.mir
pub fn outer_get(this: &Outer) -> u8 {
    inner_get(&this.inner)
}
