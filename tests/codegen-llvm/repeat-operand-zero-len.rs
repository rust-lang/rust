//@ compile-flags: -Copt-level=1 -Cno-prepopulate-passes

// This test is here to hit the `Rvalue::Repeat` case in `codegen_rvalue_operand`.
// It only applies when the resulting array is a ZST, so the test is written in
// such a way as to keep MIR optimizations from seeing that fact and removing
// the local and statement altogether. (At the time of writing, no other codegen
// test hit that code path, nor did a stage 2 build of the compiler.)

#![crate_type = "lib"]

#[repr(transparent)]
pub struct Wrapper<T, const N: usize>([T; N]);

// CHECK-LABEL: define {{.+}}do_repeat{{.+}}(i32 noundef{{( signext)?}} %x)
// CHECK-NEXT: start:
// CHECK-NOT: alloca
// CHECK-NEXT: ret void
#[inline(never)]
pub fn do_repeat<T: Copy, const N: usize>(x: T) -> Wrapper<T, N> {
    Wrapper([x; N])
}

// CHECK-LABEL: @trigger_repeat_zero_len
#[no_mangle]
pub fn trigger_repeat_zero_len() -> Wrapper<u32, 0> {
    // CHECK: call void {{.+}}do_repeat{{.+}}(i32 noundef{{( signext)?}} 4)
    do_repeat(4)
}
