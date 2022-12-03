// skip-filecheck
// unit-test: InstCombine
#![crate_type = "lib"]

pub struct Outer {
    inner: Inner,
}

struct Inner {
    field: u8,
}

// EMIT_MIR place_projection.place_projection.InstCombine.diff
pub fn place_projection(o: Outer) -> u8 {
    let temp = o.inner;
    temp.field
}
