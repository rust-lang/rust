use std::borrow::Cow;

use crate::spec::{
    BinaryFormat, Cc, DebuginfoKind, LinkerFlavor, Lld, SplitDebuginfo, TargetOptions, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    // We cannot use `-nodefaultlibs` because compiler-rt has to be passed
    // as a path since it's not added to linker search path by the default.
    // There were attempts to make it behave like libgcc (so one can just use -l<name>)
    // but LLVM maintainers rejected it: https://reviews.llvm.org/D51440
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        &["-nolibc", "--unwindlib=none"],
    );
    // Order of `late_link_args*` does not matter with LLD.
    let late_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        &["-lmingw32", "-lmingwex", "-lmsvcrt", "-lkernel32", "-luser32"],
    );

    TargetOptions {
        os: "windows".into(),
        env: "gnu".into(),
        vendor: "pc".into(),
        abi: "llvm".into(),
        linker: Some("clang".into()),
        dynamic_linking: true,
        dll_tls_export: false,
        dll_prefix: "".into(),
        dll_suffix: ".dll".into(),
        exe_suffix: ".exe".into(),
        families: cvs!["windows"],
        is_like_windows: true,
        binary_format: BinaryFormat::Coff,
        allows_weak_linkage: false,
        pre_link_args,
        late_link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        eh_frame_header: false,
        no_default_libraries: false,
        has_thread_local: true,
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        debuginfo_kind: DebuginfoKind::Dwarf,
        // FIXME(davidtwco): Support Split DWARF on Windows GNU - may require LLVM changes to
        // output DWO, despite using DWARF, doesn't use ELF..
        supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
        ..Default::default()
    }
}
