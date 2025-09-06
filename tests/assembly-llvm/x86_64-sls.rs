// Test harden-sls flag

#![feature(core_intrinsics)]
//@ revisions: NONE ALL RET IJMP
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -Cunsafe-allow-abi-mismatch=harden-sls
//@ [NONE] compile-flags: -Zharden-sls=none
//@ [ALL] compile-flags: -Zharden-sls=all
//@ [RET] compile-flags: -Zharden-sls=return
//@ [IJMP] compile-flags: -Zharden-sls=indirect-jmp
//@ only-x86_64
#![crate_type = "lib"]

#[no_mangle]
pub fn double_return(a: i32, b: i32) -> i32 {
    // CHECK-LABEL: double_return:
    // CHECK:         jle
    // CHECK-NOT:     int3
    // CHECK:         retq
    // RET-NEXT:      int3
    // ALL-NEXT:      int3
    // IJMP-NOT:      int3
    // NONE-NOT:      int3
    // CHECK:         retq
    // RET-NEXT:      int3
    // ALL-NEXT:      int3
    // IJMP-NOT:      int3
    // NONE-NOT:      int3
    if a > 0 {
        unsafe { std::intrinsics::unchecked_div(a, b) }
    } else {
        unsafe { std::intrinsics::unchecked_div(b, a) }
    }
}

#[no_mangle]
pub fn indirect_branch(a: i32, b: i32, i: i32) -> i32 {
    // CHECK-LABEL: indirect_branch:
    // CHECK:         jmpq *
    // RET-NOT:       int3
    // NONE-NOT:      int3
    // IJMP-NEXT:     int3
    // ALL-NEXT:      int3
    // CHECK:         retq
    // RET-NEXT:      int3
    // ALL-NEXT:      int3
    // IJMP-NOT:      int3
    // NONE-NOT:      int3
    // CHECK:         retq
    // RET-NEXT:      int3
    // ALL-NEXT:      int3
    // IJMP-NOT:      int3
    // NONE-NOT:      int3
    match i {
        0 => unsafe { std::intrinsics::unchecked_div(a, b) },
        1 => unsafe { std::intrinsics::unchecked_div(b, a) },
        2 => unsafe { std::intrinsics::unchecked_div(b, a) + 2 },
        3 => unsafe { std::intrinsics::unchecked_div(b, a) + 3 },
        4 => unsafe { std::intrinsics::unchecked_div(b, a) + 4 },
        5 => unsafe { std::intrinsics::unchecked_div(b, a) + 5 },
        6 => unsafe { std::intrinsics::unchecked_div(b, a) + 6 },
        _ => panic!(""),
    }
}

#[no_mangle]
pub fn bar(ptr: fn()) {
    // CHECK-LABEL: bar:
    // CHECK:         jmpq *
    // RET-NOT:       int3
    // NONE-NOT:      int3
    // IJMP-NEXT:     int3
    // ALL-NEXT:      int3
    // CHECK-NOT:     ret
    ptr()
}
