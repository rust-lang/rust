use rustc_abi::Endian;

use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, RelocModel, Target, TargetMetadata, TargetOptions, cvs,
};

/// A base target for PlayStation Vita devices using the VITASDK toolchain (using newlib).
///
/// Requires the VITASDK toolchain on the host system.

pub(crate) fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        &["-Wl,-q", "-Wl,--pic-veneer"],
    );

    Target {
        llvm_target: "thumbv7a-sony-vita-eabihf".into(),
        metadata: TargetMetadata {
            description: Some(
                "Armv7-A Cortex-A9 Sony PlayStation Vita (requires VITASDK toolchain)".into(),
            ),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "vita".into(),
            endian: Endian::Little,
            c_int_width: 32,
            env: "newlib".into(),
            vendor: "sony".into(),
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            no_default_libraries: false,
            cpu: "cortex-a9".into(),
            families: cvs!["unix"],
            linker: Some("arm-vita-eabi-gcc".into()),
            relocation_model: RelocModel::Static,
            features: "+v7,+neon,+vfp3,+thumb2,+thumb-mode".into(),
            pre_link_args,
            exe_suffix: ".elf".into(),
            has_thumb_interworking: true,
            max_atomic_width: Some(64),
            ..Default::default()
        },
    }
}
