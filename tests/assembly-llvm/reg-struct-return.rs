//! Tests that -Zreg-struct-return changes ABI for small struct returns
//! from hidden-pointer convention to register-return on x86_32.
//! This test covers:
//! * Direct struct construction, verifying register return versus hidden pointer
//! * External function calls returning structs, verifying ABI mismatch handling
//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -O --target=i686-unknown-linux-gnu -Crelocation-model=static
//@ revisions: WITH WITHOUT
//@[WITH] compile-flags: -Zreg-struct-return
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Verifies ABI changes for small struct, where both fields fit into one register.
// WITH is expected to use register return, WITHOUT should use hidden pointer.
mod small {
    struct small_t {
        a: i8,
        b: i8,
    }

    unsafe extern "C" {
        fn small() -> small_t;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn small_direct_construction() -> small_t {
        // (42 << 8) | 42 = 10794

        // WITH-LABEL: small_direct_construction
        // WITH: movw $10794, %ax
        // WITH: retl

        // WITHOUT-LABEL: small_direct_construction
        // WITHOUT: movl 4(%esp), %e{{.*}}
        // WITHOUT: movw $10794, (%e{{.*}})
        // WITHOUT: retl $4
        small_t { a: 42, b: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn small_call() -> small_t {
        // WITH-LABEL: small_call
        // WITH: jmp small

        // WITHOUT-LABEL: small_call
        // WITHOUT: calll small
        // WITHOUT: retl $4
        small()
    }
}

// Verifies ABI changes for a struct of size 8, which is the maximum size
// for reg-struct-return.
// WITH is expected to still use register return, WITHOUT should use hidden
// pointer.
mod pivot {
    struct pivot_t {
        a: i32,
        b: i32,
    }

    unsafe extern "C" {
        fn pivot() -> pivot_t;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pivot_direct_construction() -> pivot_t {
        // WITH-LABEL: pivot_direct_construction
        // WITH: movl $42, %e{{.*}}
        // WITH: movl $42, %e{{.*}}
        // WITH: retl

        // WITHOUT-LABEL: pivot_direct_construction
        // WITHOUT: movl 4(%esp), %e{{.*}}
        // WITHOUT: movl $42, (%e{{.*}})
        // WITHOUT: movl $42, 4(%e{{.*}})
        // WITHOUT: retl $4
        pivot_t { a: 42, b: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pivot_call() -> pivot_t {
        // WITH-LABEL: pivot_call
        // WITH: jmp pivot

        // WITHOUT-LABEL: pivot_call
        // WITHOUT: calll pivot
        // WITHOUT: retl $4
        pivot()
    }
}

// Verifies ABI changes for a struct of size 12, which is larger than the
// maximum size for reg-struct-return (8 bytes).
// Here, both WITH and WITHOUT should use the hidden pointer convention.
mod large {
    struct large_t {
        a: i32,
        b: i32,
        c: i32,
    }

    unsafe extern "C" {
        fn large() -> large_t;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn large_direct_construction() -> large_t {
        // WITH-LABEL: large_direct_construction
        // WITH: movl 4(%esp), %e{{.*}}
        // WITH: movl $42, (%e{{.*}})
        // WITH: movl $42, 4(%e{{.*}})
        // WITH: movl $42, 8(%e{{.*}})
        // WITH: retl $4

        // WITHOUT-LABEL: large_direct_construction
        // WITHOUT: movl 4(%esp), %e{{.*}}
        // WITHOUT: movl $42, (%e{{.*}})
        // WITHOUT: movl $42, 4(%e{{.*}})
        // WITHOUT: movl $42, 8(%e{{.*}})
        // WITHOUT: retl $4
        large_t { a: 42, b: 42, c: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn large_call() -> large_t {
        // WITH-LABEL: large_call
        // WITH: calll large
        // WITH: retl $4

        // WITHOUT-LABEL: large_call
        // WITHOUT: calll large
        // WITHOUT: retl $4
        large()
    }
}
