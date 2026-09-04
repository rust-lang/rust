//@ add-minicore
//
//@ revisions: POWERPC POWERPC64LE POWERPC64 AIX
//@ [POWERPC] compile-flags: --target powerpc-unknown-linux-gnu
//@ [POWERPC64LE] compile-flags: --target powerpc64le-unknown-linux-gnu
//@ [POWERPC64] compile-flags: --target powerpc64-unknown-linux-gnu
//@ [AIX] compile-flags: --target powerpc64-ibm-aix
//@ compile-flags: -Copt-level=3 --crate-type=lib
//@ needs-llvm-components: powerpc

#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
#[cfg(target_arch = "powerpc")]
use minicore::arch::powerpc::ppcf128;
#[cfg(target_arch = "powerpc64")]
use minicore::arch::powerpc64::ppcf128;
use minicore::*;

/// On elfv1 and aix single-float structs are passed as scalar arguments.
#[repr(C)]
struct Hfa1 {
    a: ppcf128,
}

/// On elfv2 homogenous aggregates of up to 4 elements are passed as scalars.
#[repr(C)]
struct Hfa2 {
    a: ppcf128,
    b: ppcf128,
}

#[repr(C)]
struct Hfa4 {
    a: ppcf128,
    b: ppcf128,
    c: ppcf128,
    d: ppcf128,
}

#[repr(C)]
struct NonHfa5 {
    a: ppcf128,
    b: ppcf128,
    c: ppcf128,
    d: ppcf128,
    e: ppcf128,
}

// CHECK-LABEL: ppc_fp128 @scalar_second(ppc_fp128 noundef %_a, ppc_fp128 noundef returned %b)
#[unsafe(no_mangle)]
extern "C" fn scalar_second(_a: ppcf128, b: ppcf128) -> ppcf128 {
    // CHECK: ret ppc_fp128 %b
    b
}

// POWERPC64-LABEL: void @hfa1(ptr {{.*}}sret([16 x i8]) {{.*}}, ppc_fp128 %0)
// POWERPC64LE-LABEL: ppc_fp128 @hfa1(ppc_fp128 returned %0)
// AIX-LABEL: void @hfa1(ptr {{.*}}sret([16 x i8]) {{.*}}, ptr {{.*}}byval([16 x i8]) {{.*}})
// POWER-LABEL: void @hfa1(ptr {{.*}}sret([16 x i8]) {{.*}}, ptr {{.*}}byval([16 x i8]) {{.*}})
#[unsafe(no_mangle)]
extern "C" fn hfa1(x: Hfa1) -> Hfa1 {
    x
}

// POWERPC64-LABEL: ppc_fp128 @hfa2([2 x i128] %0)
// POWERPC64LE-LABEL: ppc_fp128 @hfa2([2 x ppc_fp128] %0)
// AIX-LABEL: ppc_fp128 @hfa2(ptr {{.*}}byval([32 x i8])
// POWER-LABEL: ppc_fp128 @hfa2(ptr {{.*}}byval([32 x i8])
#[unsafe(no_mangle)]
extern "C" fn hfa2(x: Hfa2) -> ppcf128 {
    x.b
}

// POWERPC64-LABEL: ppc_fp128 @hfa4([4 x i128] %0)
// POWERPC64LE-LABEL: ppc_fp128 @hfa4([4 x ppc_fp128] %0)
// AIX-LABEL: ppc_fp128 @hfa4(ptr {{.*}}byval([64 x i8])
// POWER-LABEL: ppc_fp128 @hfa4(ptr {{.*}}byval([64 x i8])
#[unsafe(no_mangle)]
extern "C" fn hfa4(x: Hfa4) -> ppcf128 {
    x.d
}

// POWERPC64-LABEL: ppc_fp128 @non_hfa5([5 x i128] %0)
// POWERPC64LE-LABEL: ppc_fp128 @non_hfa5([5 x i128] %0)
// AIX-LABEL: ppc_fp128 @non_hfa5(ptr {{.*}}byval([80 x i8])
// POWERPC-LABEL: ppc_fp128 @non_hfa5(ptr {{.*}}dereferenceable(80)
#[unsafe(no_mangle)]
extern "C" fn non_hfa5(x: NonHfa5) -> ppcf128 {
    x.c
}
