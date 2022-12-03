// compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2
// only-64bit
// ignore-debug

#![crate_type = "lib"]

// EMIT_MIR simple_swap.simple_swap.PreCodegen.after.mir
pub fn simple_swap<T>(x: &mut T, y: &mut T) {
    use std::ptr::{read, write};
    unsafe {
        let temp = read(x);
        write(x, read(y));
        write(y, temp);
    }
}
