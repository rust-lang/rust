use crate::spec::{FramePointer, RelroLevel, TargetOptions, TlsModel, cvs};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "openbsd".into(),
        dynamic_linking: true,
        families: cvs!["unix"],
        has_rpath: true,
        abi_return_struct_as_int: true,
        position_independent_executables: true,
        frame_pointer: FramePointer::Always, // FIXME 43575: should be MayOmit...
        relro_level: RelroLevel::Full,
        default_dwarf_version: 2,
        tls_model: TlsModel::Emulated,
        ..Default::default()
    }
}
