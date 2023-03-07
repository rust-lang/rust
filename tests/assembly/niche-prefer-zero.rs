// Check that niche selection prefers zero and that jumps are optimized away.
// See https://github.com/rust-lang/rust/pull/87794
// assembly-output: emit-asm
// only-x86
// compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(u8)]
pub enum Size {
    One = 1,
    Two = 2,
    Three = 3,
}

#[no_mangle]
pub fn handle(x: Option<Size>) -> u8 {
    match x {
        None => 0,
        Some(size) => size as u8,
    }
}

// There should be no jumps in output
// CHECK-NOT: j
