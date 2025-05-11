use rustc_abi::Endian;

use crate::spec::{
    BinaryFormat, Cc, CodeModel, LinkOutputKind, LinkerFlavor, TargetOptions, crt_objects, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        abi: "vec-extabi".into(),
        code_model: Some(CodeModel::Large),
        cpu: "pwr7".into(),
        os: "aix".into(),
        vendor: "ibm".into(),
        dynamic_linking: true,
        endian: Endian::Big,
        executables: true,
        archive_format: "aix_big".into(),
        families: cvs!["unix"],
        has_rpath: false,
        has_thread_local: true,
        crt_static_respected: true,
        linker_flavor: LinkerFlavor::Unix(Cc::No),
        linker: Some("ld".into()),
        eh_frame_header: false,
        is_like_aix: true,
        binary_format: BinaryFormat::Xcoff,
        default_dwarf_version: 3,
        function_sections: true,
        pre_link_objects: crt_objects::new(&[
            (LinkOutputKind::DynamicNoPicExe, &["/usr/lib/crt0_64.o", "/usr/lib/crti_64.o"]),
            (LinkOutputKind::DynamicPicExe, &["/usr/lib/crt0_64.o", "/usr/lib/crti_64.o"]),
        ]),
        dll_suffix: ".a".into(),
        ..Default::default()
    }
}
