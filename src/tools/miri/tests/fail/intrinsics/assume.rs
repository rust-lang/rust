#![feature(core_intrinsics)]

fn main() {
    let x = 5;
    unsafe {
        std::intrinsics::assume(x < 10);
        std::intrinsics::assume(x > 1);
        std::intrinsics::assume(x > 42); //~ ERROR: `assume` called with `false`
    }
}
