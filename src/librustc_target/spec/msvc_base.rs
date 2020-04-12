// https://docs.microsoft.com/en-us/cpp/build/reference/linking
// https://docs.microsoft.com/en-us/cpp/build/reference/libpath-additional-libpath
// > LINK first processes options specified in the LINK environment variable, followed by options
// > in the order they are specified on the command line and in command files.
// > If an option is repeated with different arguments, the last one processed takes precedence.
// > Options apply to the entire build; no options can be applied to specific input files.
// > If you want to specify more than one directory, you must specify multiple /LIBPATH options.
// > The linker will then search the specified directories in order.
//
// Therefore all options that are not input files are order-independent and either non-overridable
// or right-overridable. Library search directories are left-overridable.

use crate::spec::{LinkArgsMap, LinkerFlavor, LldFlavor, NewLinkArgs, TargetOptions};

pub fn opts() -> TargetOptions {
    let new_link_args = NewLinkArgs {
        unordered_non_overridable: vec![
            // Suppress the verbose logo and authorship debugging output, which would needlessly
            // clog any log files.
            "/NOLOGO".to_string(),
        ],
        unordered_right_overridable: vec![
            // Tell the compiler that non-code sections can be marked as non-executable,
            // including stack pages.
            // UEFI is fully compatible to non-executable data pages.
            // In fact, firmware might enforce this, so we better let the linker know about this,
            // so it will fail if the compiler ever tries placing code on the stack
            // (e.g., trampoline constructs and alike).
            "/NXCOMPAT".to_string(),
        ],
        ..Default::default()
    };

    let mut link_args = LinkArgsMap::new();
    link_args.insert(LinkerFlavor::Msvc, new_link_args.clone());
    link_args.insert(LinkerFlavor::Lld(LldFlavor::Link), new_link_args);

    TargetOptions {
        executables: true,
        is_like_windows: true,
        is_like_msvc: true,
        // set VSLANG to 1033 can prevent link.exe from using
        // language packs, and avoid generating Non-UTF-8 error
        // messages if a link error occurred.
        link_env: vec![("VSLANG".to_string(), "1033".to_string())],
        lld_flavor: LldFlavor::Link,
        link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,

        ..Default::default()
    }
}
