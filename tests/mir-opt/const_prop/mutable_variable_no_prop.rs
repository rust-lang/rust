// skip-filecheck
// unit-test: ConstProp

static mut STATIC: u32 = 0x42424242;

// EMIT_MIR mutable_variable_no_prop.main.ConstProp.diff
fn main() {
    let mut x = 42;
    unsafe {
        x = STATIC;
    }
    let y = x;
}
