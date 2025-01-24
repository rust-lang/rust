use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    // In QNX, libc does not provide a compatible ABI between versions.
    // To distinguish between QNX versions, we needed a stable conditional compilation switch,
    // which is why we needed to implement different targets in the compiler.
    Target {
        llvm_target: "aarch64-unknown-unknown".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARM64 QNX Neutrino 7.0 RTOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        // from: https://llvm.org/docs/LangRef.html#data-layout
        // e         = little endian
        // m:e       = ELF mangling: Private symbols get a .L prefix
        // i8:8:32   = 8-bit-integer, minimum_alignment=8, preferred_alignment=32
        // i16:16:32 = 16-bit-integer, minimum_alignment=16, preferred_alignment=32
        // i64:64    = 64-bit-integer, minimum_alignment=64, preferred_alignment=64
        // i128:128  = 128-bit-integer, minimum_alignment=128, preferred_alignment=128
        // n32:64    = 32 and 64 are native integer widths; Elements of this set are considered to support most general arithmetic operations efficiently.
        // S128      = 128 bits are the natural alignment of the stack in bits.
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a".into(),
            max_atomic_width: Some(128),
            pre_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[
                "-Vgcc_ntoaarch64le_cxx",
            ]),
            env: "nto70".into(),
            ..base::nto_qnx::opts()
        },
    }
}
