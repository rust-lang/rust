#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

// EMIT_MIR pattern_types.main.PreCodegen.after.mir
fn main() {
    // CHECK: debug x => const 2_u32 is 1..
    let x: pattern_type!(u32 is 1..) = unsafe { std::mem::transmute(2) };
    // CHECK: debug y => const {transmute(0x00000000): (u32) is 1..}
    let y: pattern_type!(u32 is 1..) = unsafe { std::mem::transmute(0) };
}
