// skip-filecheck
// unit-test: ConstProp

// The struct has scalar ABI, but is not a scalar type.
// Make sure that we handle this correctly.
#[repr(transparent)]
struct I32(i32);

// EMIT_MIR repr_transparent.main.ConstProp.diff
fn main() {
    let x = I32(0);
    let y = I32(x.0 + x.0);
}
