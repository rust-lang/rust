use crate::spec::{cvs, LinkArgs, LinkerFlavor, RelocModel, Target, TargetOptions};

/// A base target for Nintendo 3DS devices using the devkitARM toolchain.
///
/// Requires the devkitARM toolchain for 3DS targets on the host system.

pub fn target() -> Target {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-specs=3dsx.specs".into(),
            "-mtune=mpcore".into(),
            "-mfloat-abi=hard".into(),
            "-mtp=soft".into(),
        ],
    );

    Target {
        llvm_target: "armv6k-none-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "horizon".into(),
            env: "newlib".into(),
            vendor: "nintendo".into(),
            abi: "eabihf".into(),
            linker_flavor: LinkerFlavor::Gcc,
            cpu: "mpcore".into(),
            executables: true,
            families: cvs!["unix"],
            linker: Some("arm-none-eabi-gcc".into()),
            relocation_model: RelocModel::Static,
            features: "+vfp2".into(),
            pre_link_args,
            exe_suffix: ".elf".into(),
            no_default_libraries: false,
            // There are some issues in debug builds with this enabled in certain programs.
            has_thread_local: false,
            ..Default::default()
        },
    }
}
