//@ compile-flags: -O -Zmir-opt-level=2

#![crate_type = "lib"]
#![feature(transmute_neo)]
#![feature(transmute_prefix)]

use std::mem::{transmute_neo, transmute_prefix};

// EMIT_MIR transmutes.neo_to_cast.PreCodegen.after.mir
pub fn neo_to_cast(x: f32) -> i32 {
    // CHECK-LABEL: fn neo_to_cast
    // CHECK: _0 = copy _1 as i32 (Transmute);
    unsafe { transmute_neo(x) }
}

// EMIT_MIR transmutes.prefix_to_cast.PreCodegen.after.mir
pub fn prefix_to_cast(x: f32) -> i32 {
    // CHECK-LABEL: fn prefix_to_cast
    // CHECK: _0 = copy _1 as i32 (Transmute);
    unsafe { transmute_prefix(x) }
}

// EMIT_MIR transmutes.prefix_of_array.PreCodegen.after.mir
pub fn prefix_of_array(x: [u32; 4]) -> [u32; 2] {
    // CHECK-LABEL: fn prefix_of_array
    // CHECK: _2 = copy _1 as {{.+}}::Transmute<[u32; 4], [u32; 2]> (Transmute);
    // CHECK: _3 = move (_2.1: {{.+}}::ManuallyDrop<[u32; 2]>);
    // CHECK: _0 = copy _3 as [u32; 2] (Transmute);
    unsafe { transmute_prefix(x) }
}

#[repr(C, align(64))]
struct Align64<T>(T);

// EMIT_MIR transmutes.pad_for_alignment.PreCodegen.after.mir
pub fn pad_for_alignment(x: u32) -> Align64<u32> {
    // CHECK-LABEL: fn pad_for_alignment
    // CHECK: _2 = copy _1 as {{.+}}::ManuallyDrop<u32> (Transmute);
    // CHECK: _3 = {{.+}}::Transmute::<u32, Align64<u32>> { a: copy _2 };
    // CHECK: _0 = move _3 as Align64<u32> (Transmute);
    unsafe { transmute_prefix(x) }
}

// EMIT_MIR transmutes.forget_at_home.PreCodegen.after.mir
pub fn forget_at_home(x: String) {
    // CHECK-LABEL: fn forget_at_home
    // CHECK: bb0:
    // CHECK-NEXT: return;
    unsafe { transmute_prefix(x) }
}
