use crate::abi::Endian;
use crate::spec::{crt_objects, cvs, Cc, CodeModel, LinkOutputKind, LinkerFlavor};
use crate::spec::{MaybeLazy, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        abi: "vec-extabi".into(),
        code_model: Some(CodeModel::Small),
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
        default_dwarf_version: 3,
        function_sections: true,
        pre_link_objects: MaybeLazy::lazy(|| {
            crt_objects::new(&[
                (LinkOutputKind::DynamicNoPicExe, &["/usr/lib/crt0_64.o", "/usr/lib/crti_64.o"]),
                (LinkOutputKind::DynamicPicExe, &["/usr/lib/crt0_64.o", "/usr/lib/crti_64.o"]),
            ])
        }),
        dll_suffix: ".a".into(),
        ..Default::default()
    }
}
