//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib
//@ only-nvptx64
//@ ignore-nvptx64

#![feature(abi_ptx, core_intrinsics)]
#![no_std]

use core::intrinsics::*;

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Currently, LLVM NVPTX backend can only emit atomic instructions with
// `relaxed` (PTX default) ordering. But it's also useful to make sure
// the backend won't fail with other orders. Apparently, the backend
// doesn't support fences as well. As a workaround `llvm.nvvm.membar.*`
// could work, and perhaps on the long run, all the atomic operations
// should rather be provided by `core::arch::nvptx`.

// Also, PTX ISA doesn't have atomic `load`, `store` and `nand`.

// FIXME(denzp): add tests for `core::sync::atomic::*`.

#[no_mangle]
pub unsafe extern "ptx-kernel" fn atomics_kernel(a: *mut u32) {
    // CHECK: atom.global.and.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.and.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_and(a, 1);
    atomic_and_relaxed(a, 1);

    // CHECK: atom.global.cas.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1, 2;
    // CHECK: atom.global.cas.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1, 2;
    atomic_cxchg(a, 1, 2);
    atomic_cxchg_relaxed(a, 1, 2);

    // CHECK: atom.global.max.s32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.max.s32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_max(a, 1);
    atomic_max_relaxed(a, 1);

    // CHECK: atom.global.min.s32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.min.s32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_min(a, 1);
    atomic_min_relaxed(a, 1);

    // CHECK: atom.global.or.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.or.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_or(a, 1);
    atomic_or_relaxed(a, 1);

    // CHECK: atom.global.max.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.max.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_umax(a, 1);
    atomic_umax_relaxed(a, 1);

    // CHECK: atom.global.min.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.min.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_umin(a, 1);
    atomic_umin_relaxed(a, 1);

    // CHECK: atom.global.add.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.add.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_xadd(a, 1);
    atomic_xadd_relaxed(a, 1);

    // CHECK: atom.global.exch.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.exch.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_xchg(a, 1);
    atomic_xchg_relaxed(a, 1);

    // CHECK: atom.global.xor.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    // CHECK: atom.global.xor.b32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], 1;
    atomic_xor(a, 1);
    atomic_xor_relaxed(a, 1);

    // CHECK: mov.u32 %[[sub_0_arg:r[0-9]+]], 100;
    // CHECK: neg.s32 temp, %[[sub_0_arg]];
    // CHECK: atom.global.add.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], temp;
    atomic_xsub(a, 100);

    // CHECK: mov.u32 %[[sub_1_arg:r[0-9]+]], 200;
    // CHECK: neg.s32 temp, %[[sub_1_arg]];
    // CHECK: atom.global.add.u32 %{{r[0-9]+}}, [%{{rd[0-9]+}}], temp;
    atomic_xsub_relaxed(a, 200);
}
