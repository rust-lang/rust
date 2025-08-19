use std::borrow::Cow;

use crate::spec::{
    BinaryFormat, Cc, DebuginfoKind, LinkerFlavor, Lld, SplitDebuginfo, TargetOptions, TlsModel,
    cvs,
};

pub(crate) fn opts() -> TargetOptions {
    let mut pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &["--disable-dynamicbase", "--enable-auto-image-base"],
    );
    crate::spec::add_link_args(
        &mut pre_link_args,
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        &["-Wl,--disable-dynamicbase", "-Wl,--enable-auto-image-base"],
    );
    let cygwin_libs = &["-lcygwin", "-lgcc", "-lcygwin", "-luser32", "-lkernel32", "-lgcc_s"];
    let mut late_link_args =
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), cygwin_libs);
    crate::spec::add_link_args(
        &mut late_link_args,
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        cygwin_libs,
    );
    TargetOptions {
        os: "cygwin".into(),
        vendor: "pc".into(),
        // FIXME(#13846) this should be enabled for cygwin
        function_sections: false,
        linker: Some("gcc".into()),
        dynamic_linking: true,
        dll_prefix: "".into(),
        dll_suffix: ".dll".into(),
        exe_suffix: ".exe".into(),
        families: cvs!["unix"],
        is_like_windows: true,
        binary_format: BinaryFormat::Coff,
        allows_weak_linkage: false,
        pre_link_args,
        late_link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        requires_uwtable: true,
        eh_frame_header: false,
        debuginfo_kind: DebuginfoKind::Dwarf,
        supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
        tls_model: TlsModel::Emulated,
        has_thread_local: true,
        ..Default::default()
    }
}
