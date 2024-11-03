// This defines the ia32 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options. On ia32 systems
// UEFI systems always run in protected-mode, have the interrupt-controller pre-configured and
// force a single-CPU execution.
// The cdecl ABI is used. It differs from the stdcall or fastcall ABI.

use crate::spec::{Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::uefi_msvc::opts();
    base.cpu = "pentium4".into();
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
    base.features = "-mmx,-sse,+soft-float".into();

    Target {
        llvm_target: "i686-unknown-uefi".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("32-bit UEFI".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),

        options: base,
    }
}
