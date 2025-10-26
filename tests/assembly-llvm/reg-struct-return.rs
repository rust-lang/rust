//! Tests that -Zreg-struct-return changes ABI for small struct returns
//! from hidden-pointer convention to register-return on x86_32.
//! This test covers:
//! * Callee side, verifying that the structs are correctly loaded into registers when
//!   `-Zreg-struct-return` is activated
//! * Caller side, verifying callers do receive returned structs in registers when
//!   `-Zreg-struct-return` is activated
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

// Verifies ABI changes for small structs, where both fields fit into one register.
// WITH is expected to use register return, WITHOUT should use hidden pointer.
mod Small {
    struct SmallStruct {
        a: i8,
        b: i8,
    }

    unsafe extern "C" {
        fn small() -> SmallStruct;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn small_callee() -> SmallStruct {
        // (42 << 8) | 42 = 10794

        // WITH-LABEL: small_callee
        // WITH: movw $10794, %ax
        // WITH: retl

        // WITHOUT-LABEL: small_callee
        // WITHOUT: movl 4(%esp), %e{{.*}}
        // WITHOUT: movw $10794, (%e{{.*}})
        // WITHOUT: retl $4
        SmallStruct { a: 42, b: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn small_caller(dst: &mut SmallStruct) {
        // WITH-LABEL: small_caller
        // WITH: calll small
        // WITH: movw %ax, (%e{{.*}})

        // WITHOUT-LABEL: small_caller
        // WITHOUT: calll small
        // WITHOUT: movzwl {{.*}}(%esp), %e[[TMP:..]]
        // WITHOUT: movw %[[TMP]], (%e{{..}})
        *dst = small();
    }
}

// Verifies ABI changes for a struct of size 8, which is the maximum size
// for reg-struct-return.
// WITH is expected to still use register return, WITHOUT should use hidden
// pointer.
mod Pivot {
    struct PivotStruct {
        a: i32,
        b: i32,
    }

    unsafe extern "C" {
        fn pivot() -> PivotStruct;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pivot_callee() -> PivotStruct {
        // WITH-LABEL: pivot_callee
        // WITH: movl $42, %e{{.*}}
        // WITH: movl $42, %e{{.*}}
        // WITH: retl

        // WITHOUT-LABEL: pivot_callee
        // WITHOUT: movl 4(%esp), %e{{.*}}
        // WITHOUT-DAG: movl $42, (%e{{.*}})
        // WITHOUT-DAG: movl $42, 4(%e{{.*}})
        // WITHOUT: retl $4
        PivotStruct { a: 42, b: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn pivot_caller(dst: &mut PivotStruct) {
        // WITH-LABEL: pivot_caller
        // WITH: calll pivot
        // WITH-DAG: movl %e{{.*}}, 4(%e{{.*}})
        // WITH-DAG: movl %e{{.*}}, (%e{{.*}})

        // WITHOUT-LABEL: pivot_caller
        // WITHOUT: calll pivot
        // WITHOUT: movsd {{.*}}(%esp), %[[TMP:xmm.]]
        // WITHOUT: movsd %[[TMP]], (%e{{..}})
        *dst = pivot();
    }
}

// Verifies ABI changes for a struct of size 12, which is larger than the
// maximum size for reg-struct-return (8 bytes).
// Here, the hidden pointer convention should be used even when `-Zreg-struct-return` is set.
mod Large {
    struct LargeStruct {
        a: i32,
        b: i32,
        c: i32,
    }

    unsafe extern "C" {
        fn large() -> LargeStruct;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn large_callee() -> LargeStruct {
        // CHECK-LABEL: large_callee
        // CHECK: movl 4(%esp), %e{{.*}}
        // CHECK-DAG: movl $42, (%e{{.*}})
        // CHECK-DAG: movl $42, 4(%e{{.*}})
        // CHECK-DAG: movl $42, 8(%e{{.*}})
        // CHECK: retl $4
        LargeStruct { a: 42, b: 42, c: 42 }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn large_caller(dst: &mut LargeStruct) {
        // CHECK-LABEL: large_caller
        // CHECK: calll large
        // CHECK-DAG: movl   {{.*}}(%esp), %[[TMP1:e..]]
        // CHECK-DAG: movl  %[[TMP1]], {{.*}}(%e{{..}})
        // CHECK-DAG: movsd  {{.*}}(%esp), %[[TMP2:xmm.]]
        // CHECK-DAG: movsd  %[[TMP2]], {{.*}}(%e{{..}})
        *dst = large();
    }
}
