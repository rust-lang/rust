// This defines a base target-configuration for native UEFI systems. The UEFI specification has
// quite detailed sections on the ABI of all the supported target architectures. In almost all
// cases it simply follows what Microsoft Windows does. Hence, whenever in doubt, see the MSDN
// documentation.
// UEFI uses COFF/PE32+ format for binaries. All binaries must be statically linked. No dynamic
// linker is supported. As native to COFF, binaries are position-dependent, but will be relocated
// by the loader if the pre-chosen memory location is already in use.
// UEFI forbids running code on anything but the boot-CPU. No interrupts are allowed other than
// the timer-interrupt. Device-drivers are required to use polling-based models. Furthermore, all
// code runs in the same environment, no process separation is supported.

use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy};
use crate::spec::{StackProbeType, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = super::msvc_base::opts();

    base.add_pre_link_args(
        LinkerFlavor::Msvc,
        &[
            // Non-standard subsystems have no default entry-point in PE+ files. We have to define
            // one. "efi_main" seems to be a common choice amongst other implementations and the
            // spec.
            "/entry:efi_main",
            // COFF images have a "Subsystem" field in their header, which defines what kind of
            // program it is. UEFI has 3 fields reserved, which are EFI_APPLICATION,
            // EFI_BOOT_SERVICE_DRIVER, and EFI_RUNTIME_DRIVER. We default to EFI_APPLICATION,
            // which is very likely the most common option. Individual projects can override this
            // with custom linker flags.
            // The subsystem-type only has minor effects on the application. It defines the memory
            // regions the application is loaded into (runtime-drivers need to be put into
            // reserved areas), as well as whether a return from the entry-point is treated as
            // exit (default for applications).
            "/subsystem:efi_application",
        ],
    );

    TargetOptions {
        os: "uefi".into(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Link),
        disable_redzone: true,
        exe_suffix: ".efi".into(),
        allows_weak_linkage: false,
        panic_strategy: PanicStrategy::Abort,
        // LLVM does not emit inline assembly because the LLVM target does not get considered as…
        // "Windows".
        stack_probes: StackProbeType::Call,
        singlethread: true,
        linker: Some("rust-lld".into()),
        ..base
    }
}
