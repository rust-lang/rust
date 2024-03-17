use crate::spec::{base, LinkerFlavor, Lld, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.vendor = "win7".into();

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
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
