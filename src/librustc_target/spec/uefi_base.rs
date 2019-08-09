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

use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, PanicStrategy, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut pre_link_args = LinkArgs::new();

    pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Link), vec![
            // Suppress the verbose logo and authorship debugging output, which would needlessly
            // clog any log files.
            "/NOLOGO".to_string(),

            // UEFI is fully compatible to non-executable data pages. Tell the compiler that
            // non-code sections can be marked as non-executable, including stack pages. In fact,
            // firmware might enforce this, so we better let the linker know about this, so it
            // will fail if the compiler ever tries placing code on the stack (e.g., trampoline
            // constructs and alike).
            "/NXCOMPAT".to_string(),

            // There is no runtime for UEFI targets, prevent them from being linked. UEFI targets
            // must be freestanding.
            "/nodefaultlib".to_string(),

            // Non-standard subsystems have no default entry-point in PE+ files. We have to define
            // one. "efi_main" seems to be a common choice amongst other implementations and the
            // spec.
            "/entry:efi_main".to_string(),

            // COFF images have a "Subsystem" field in their header, which defines what kind of
            // program it is. UEFI has 3 fields reserved, which are EFI_APPLICATION,
            // EFI_BOOT_SERVICE_DRIVER, and EFI_RUNTIME_DRIVER. We default to EFI_APPLICATION,
            // which is very likely the most common option. Individual projects can override this
            // with custom linker flags.
            // The subsystem-type only has minor effects on the application. It defines the memory
            // regions the application is loaded into (runtime-drivers need to be put into
            // reserved areas), as well as whether a return from the entry-point is treated as
            // exit (default for applications).
            "/subsystem:efi_application".to_string(),
        ]);

    TargetOptions {
        dynamic_linking: false,
        executables: true,
        disable_redzone: true,
        exe_suffix: ".efi".to_string(),
        allows_weak_linkage: false,
        panic_strategy: PanicStrategy::Abort,
        stack_probes: true,
        singlethread: true,
        emit_debug_gdb_scripts: false,

        linker: Some("rust-lld".to_string()),
        lld_flavor: LldFlavor::Link,
        pre_link_args,

        .. Default::default()
    }
}
