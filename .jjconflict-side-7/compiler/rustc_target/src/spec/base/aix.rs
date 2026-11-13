use rustc_abi::Endian;

use crate::spec::{
    BinaryFormat, Cc, CfgAbi, CodeModel, LinkOutputKind, LinkerFlavor, Os, TargetOptions,
    crt_objects, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        // It makes no sense to set "vec-extabi" here without also actually configuring LLVM to use
        // that ABI. Since https://github.com/llvm/llvm-project/pull/221670 this probably just needs
        // `llvm_abiname` to be set properly. If you are doing that, make sure to also:
        // - adjust the logic in `reserved_v20to31` in `asm/powerpc.rs` to allow the extra registers
        //   to be used depending on `llvm_abiname`.
        // - adjust the logic in `spec/consistency.rs` to correlate `llvm_abiname` with `cfg_abi`.
        cfg_abi: CfgAbi::VecDefault,
        code_model: Some(CodeModel::Large),
        cpu: "pwr7".into(),
        os: Os::Aix,
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
