// compile-flags: -O

static mut STATIC: u32 = 42;

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let mut x = 42;
    unsafe {
        x = STATIC;
    }
    let y = x;
}
