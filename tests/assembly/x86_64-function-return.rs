// Test that the function return is (not) converted into a jump to the thunk
// when the `-Zfunction-return={keep,thunk-extern}` flag is (not) set.

//@ revisions: unset keep thunk-extern keep-thunk-extern thunk-extern-keep
//@ assembly-output: emit-asm
//@ compile-flags: -O
//@ [keep] compile-flags: -Zfunction-return=keep
//@ [thunk-extern] compile-flags: -Zfunction-return=thunk-extern
//@ [keep-thunk-extern] compile-flags: -Zfunction-return=keep -Zfunction-return=thunk-extern
//@ [thunk-extern-keep] compile-flags: -Zfunction-return=thunk-extern -Zfunction-return=keep
//@ only-x86_64
//@ ignore-apple Symbol is called `___x86_return_thunk` (Darwin's extra underscore)
//@ ignore-sgx Tests incompatible with LVI mitigations

#![crate_type = "lib"]

// CHECK-LABEL: foo:
#[no_mangle]
pub unsafe fn foo() {
    // CHECK-UNSET: ret
    // CHECK-UNSET-NOT: jmp __x86_return_thunk
    // CHECK-KEEP: ret
    // CHECK-KEEP-NOT: jmp __x86_return_thunk
    // CHECK-THUNK-EXTERN: jmp __x86_return_thunk
    // CHECK-THUNK-EXTERN-NOT: ret
    // CHECK-KEEP-THUNK-EXTERN: jmp __x86_return_thunk
    // CHECK-KEEP-THUNK-EXTERN-NOT: ret
    // CHECK-THUNK-EXTERN-KEEP: ret
    // CHECK-THUNK-EXTERN-KEEP-NOT: jmp __x86_return_thunk
}
