// unit-test
// compile-flags: -O

static mut STATIC: u32 = 42;

// EMIT_MIR mutable_variable_no_prop.main.ConstProp.diff
fn main() {
    let mut x = 42;
    unsafe {
        x = STATIC;
    }
    let y = x;
}
