use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let pre_link_args_msvc = vec![
        // Suppress the verbose logo and authorship debugging output, which would needlessly
        // clog any log files.
        "/NOLOGO".to_string(),
        // Tell the compiler that non-code sections can be marked as non-executable,
        // including stack pages.
        // UEFI is fully compatible to non-executable data pages.
        // In fact, firmware might enforce this, so we better let the linker know about this,
        // so it will fail if the compiler ever tries placing code on the stack
        // (e.g., trampoline constructs and alike).
        "/NXCOMPAT".to_string(),
    ];
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Msvc, pre_link_args_msvc.clone());
    pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Link), pre_link_args_msvc);

    TargetOptions {
        linker_flavor: LinkerFlavor::Msvc,
        executables: true,
        is_like_windows: true,
        is_like_msvc: true,
        lld_flavor: LldFlavor::Link,
        pre_link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,

        ..Default::default()
    }
}
