// Test that the function return is (not) converted into a jump to the thunk
// when the `-Zfunction-return={keep,thunk-extern}` flag is (not) set.

//@ revisions: unset keep thunk-extern keep-thunk-extern thunk-extern-keep
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
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
    // unset: ret
    // unset-NOT: jmp __x86_return_thunk
    // keep: ret
    // keep-NOT: jmp __x86_return_thunk
    // thunk-extern: jmp __x86_return_thunk
    // thunk-extern-NOT: ret
    // keep-thunk-extern: jmp __x86_return_thunk
    // keep-thunk-extern-NOT: ret
    // thunk-extern-keep: ret
    // thunk-extern-keep-NOT: jmp __x86_return_thunk
}
