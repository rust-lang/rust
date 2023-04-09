// ignore-wasm32 compiled with panic=abort by default
// compile-flags: -Copt-level=0 -Coverflow-checks=yes

// Tests that division with a const does not emit a panicking branch for overflow

// EMIT_MIR div_overflow.const_divisor.PreCodegen.after.mir
pub fn const_divisor(a: i32) -> i32 {
    a / 256
}

// EMIT_MIR div_overflow.const_dividend.PreCodegen.after.mir
pub fn const_dividend(a: i32) -> i32 {
    256 / a
}

fn main() {
    const_divisor(123);
    const_dividend(123);
}
