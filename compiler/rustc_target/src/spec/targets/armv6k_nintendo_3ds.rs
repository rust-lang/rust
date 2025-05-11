use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, RelocModel, Target, TargetMetadata, TargetOptions, cvs,
};

/// A base target for Nintendo 3DS devices using the devkitARM toolchain.
///
/// Requires the devkitARM toolchain for 3DS targets on the host system.

pub(crate) fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::Yes, Lld::No),
        &["-specs=3dsx.specs", "-mtune=mpcore", "-mfloat-abi=hard", "-mtp=soft"],
    );

    Target {
        llvm_target: "armv6k-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Armv6K Nintendo 3DS, Horizon (Requires devkitARM toolchain)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "horizon".into(),
            env: "newlib".into(),
            vendor: "nintendo".into(),
            cpu: "mpcore".into(),
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            families: cvs!["unix"],
            linker: Some("arm-none-eabi-gcc".into()),
            relocation_model: RelocModel::Static,
            features: "+vfp2".into(),
            pre_link_args,
            exe_suffix: ".elf".into(),
            no_default_libraries: false,
            has_thread_local: true,
            ..Default::default()
        },
    }
}
