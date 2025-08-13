// Test the regparm ABI with builtin and non-builtin calls
// Issue: https://github.com/rust-lang/rust/issues/145271
//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -O --target=i686-unknown-linux-gnu -Crelocation-model=static
//@ revisions: REGPARM1 REGPARM2 REGPARM3
//@[REGPARM1] compile-flags: -Zregparm=1
//@[REGPARM2] compile-flags: -Zregparm=2
//@[REGPARM3] compile-flags: -Zregparm=3
//@ needs-llvm-components: x86
#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

unsafe extern "C" {
    fn memset(p: *mut c_void, val: i32, len: usize) -> *mut c_void;
    fn non_builtin_memset(p: *mut c_void, val: i32, len: usize) -> *mut c_void;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(len: usize, ptr: *mut c_void, val: i32) -> *mut c_void {
    // REGPARM1-LABEL: entrypoint
    // REGPARM1: movl %e{{.*}}, %ecx
    // REGPARM1: pushl
    // REGPARM1: pushl
    // REGPARM1: calll memset

    // REGPARM2-LABEL: entrypoint
    // REGPARM2: movl 16(%esp), %edx
    // REGPARM2: movl %e{{.*}}, (%esp)
    // REGPARM2: movl %e{{.*}}, %eax
    // REGPARM2: calll memset

    // REGPARM3-LABEL: entrypoint
    // REGPARM3: movl %e{{.*}}, %esi
    // REGPARM3: movl %e{{.*}}, %eax
    // REGPARM3: movl %e{{.*}}, %ecx
    // REGPARM3: jmp memset
    unsafe { memset(ptr, val, len) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn non_builtin_entrypoint(
    len: usize,
    ptr: *mut c_void,
    val: i32,
) -> *mut c_void {
    // REGPARM1-LABEL: non_builtin_entrypoint
    // REGPARM1: movl %e{{.*}}, %ecx
    // REGPARM1: pushl
    // REGPARM1: pushl
    // REGPARM1: calll non_builtin_memset

    // REGPARM2-LABEL: non_builtin_entrypoint
    // REGPARM2: movl 16(%esp), %edx
    // REGPARM2: movl %e{{.*}}, (%esp)
    // REGPARM2: movl %e{{.*}}, %eax
    // REGPARM2: calll non_builtin_memset

    // REGPARM3-LABEL: non_builtin_entrypoint
    // REGPARM3: movl %e{{.*}}, %esi
    // REGPARM3: movl %e{{.*}}, %eax
    // REGPARM3: movl %e{{.*}}, %ecx
    // REGPARM3: jmp non_builtin_memset
    unsafe { non_builtin_memset(ptr, val, len) }
}
