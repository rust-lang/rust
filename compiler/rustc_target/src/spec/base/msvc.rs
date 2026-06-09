use std::borrow::Cow;

use crate::spec::{BinaryFormat, DebuginfoKind, LinkerFlavor, Lld, SplitDebuginfo, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    // Suppress the verbose logo and authorship debugging output, which would needlessly
    // clog any log files.
    let pre_link_args = TargetOptions::link_args(LinkerFlavor::Msvc(Lld::No), &["/NOLOGO"]);

    TargetOptions {
        linker_flavor: LinkerFlavor::Msvc(Lld::No),
        dll_tls_export: false,
        is_like_windows: true,
        is_like_msvc: true,
        binary_format: BinaryFormat::Coff,
        pre_link_args,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        archive_format: "coff".into(),

        // Currently this is the only supported method of debuginfo on MSVC
        // where `*.pdb` files show up next to the final artifact.
        split_debuginfo: SplitDebuginfo::Packed,
        supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Packed]),
        debuginfo_kind: DebuginfoKind::Pdb,

        ..Default::default()
    }
}
