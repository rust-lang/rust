use crate::spec::{LinkerFlavor, Lld, RustcAbi, SanitizerSet, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.vendor = "win7".into();
    base.rustc_abi = Some(RustcAbi::X86Sse2);
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.supported_sanitizers = SanitizerSet::ADDRESS;
    // On Windows 7 32-bit, the alignment characteristic of the TLS Directory
    // don't appear to be respected by the PE Loader, leading to crashes. As
    // a result, let's disable has_thread_local to make sure TLS goes through
    // the emulation layer.
    // See https://github.com/rust-lang/rust/issues/138903
    base.has_thread_local = false;

    base.add_pre_link_args(
        LinkerFlavor::Msvc(Lld::No),
        &[
            // Mark all dynamic libraries and executables as compatible with the larger 4GiB address
            // space available to x86 Windows binaries on x86_64.
            "/LARGEADDRESSAWARE",
            // Ensure the linker will only produce an image if it can also produce a table of
            // the image's safe exception handlers.
            // https://docs.microsoft.com/en-us/cpp/build/reference/safeseh-image-has-safe-exception-handlers
            "/SAFESEH",
        ],
    );

    Target {
        llvm_target: "i686-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: Some("32-bit MSVC (Windows 7+)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
