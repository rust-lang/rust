use crate::spec::{cvs, LinkArgs, LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let pre_link_args = LinkArgs::from([(
        LinkerFlavor::Gcc,
        vec![
            // We cannot use `-nodefaultlibs` because compiler-rt has to be passed
            // as a path since it's not added to linker search path by the default.
            // There were attemts to make it behave like libgcc (so one can just use -l<name>)
            // but LLVM maintainers rejected it: https://reviews.llvm.org/D51440
            "-nolibc".into(),
            "--unwindlib=none".into(),
        ],
    )]);
    let late_link_args = LinkArgs::from([(
        LinkerFlavor::Gcc,
        // Order of `late_link_args*` does not matter with LLD.
        vec![
            "-lmingw32".into(),
            "-lmingwex".into(),
            "-lmsvcrt".into(),
            "-lkernel32".into(),
            "-luser32".into(),
        ],
    )]);

    TargetOptions {
        os: "windows".into(),
        env: "gnu".into(),
        vendor: "pc".into(),
        abi: "llvm".into(),
        linker: Some("clang".into()),
        dynamic_linking: true,
        executables: true,
        dll_prefix: "".into(),
        dll_suffix: ".dll".into(),
        exe_suffix: ".exe".into(),
        families: cvs!["windows"],
        is_like_windows: true,
        allows_weak_linkage: false,
        pre_link_args,
        late_link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        eh_frame_header: false,
        no_default_libraries: false,
        has_thread_local: true,

        ..Default::default()
    }
}
