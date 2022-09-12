use crate::spec::{cvs, LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    // We cannot use `-nodefaultlibs` because compiler-rt has to be passed
    // as a path since it's not added to linker search path by the default.
    // There were attempts to make it behave like libgcc (so one can just use -l<name>)
    // but LLVM maintainers rejected it: https://reviews.llvm.org/D51440
    let pre_link_args =
        TargetOptions::link_args(LinkerFlavor::Gcc, &["-nolibc", "--unwindlib=none"]);
    // Order of `late_link_args*` does not matter with LLD.
    let late_link_args = TargetOptions::link_args(
        LinkerFlavor::Gcc,
        &["-lmingw32", "-lmingwex", "-lmsvcrt", "-lkernel32", "-luser32"],
    );

    TargetOptions {
        os: "windows".into(),
        env: "gnu".into(),
        vendor: "pc".into(),
        abi: "llvm".into(),
        linker: Some("clang".into()),
        dynamic_linking: true,
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
