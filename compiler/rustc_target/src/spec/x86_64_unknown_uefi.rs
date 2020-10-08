// This defines the amd64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options. On x86_64 systems (mostly called "x64" in the spec)
// UEFI systems always run in long-mode, have the interrupt-controller pre-configured and force a
// single-CPU execution.
// The win64 ABI is used. It differs from the sysv64 ABI, so we must use a windows target with
// LLVM. "x86_64-unknown-windows" is used to get the minimal subset of windows-specific features.

use crate::spec::{CodeModel, Target};

pub fn target() -> Target {
    let mut base = super::uefi_msvc_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);

    // We disable MMX and SSE for now, even though UEFI allows using them. Problem is, you have to
    // enable these CPU features explicitly before their first use, otherwise their instructions
    // will trigger an exception. Rust does not inject any code that enables AVX/MMX/SSE
    // instruction sets, so this must be done by the firmware. However, existing firmware is known
    // to leave these uninitialized, thus triggering exceptions if we make use of them. Which is
    // why we avoid them and instead use soft-floats. This is also what GRUB and friends did so
    // far.
    // If you initialize FP units yourself, you can override these flags with custom linker
    // arguments, thus giving you access to full MMX/SSE acceleration.
    base.features = "-mmx,-sse,+soft-float".to_string();

    // UEFI systems run without a host OS, hence we cannot assume any code locality. We must tell
    // LLVM to expect code to reference any address in the address-space. The "large" code-model
    // places no locality-restrictions, so it fits well here.
    base.code_model = Some(CodeModel::Large);

    Target {
        llvm_target: "x86_64-unknown-windows".to_string(),
        pointer_width: 64,
        data_layout: "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),

        options: base,
    }
}
