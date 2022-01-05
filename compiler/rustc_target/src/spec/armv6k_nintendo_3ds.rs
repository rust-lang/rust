use crate::spec::{LinkArgs, LinkerFlavor, RelocModel, Target, TargetOptions};

/// A base target for Nintendo 3DS devices using the devkitARM toolchain.
///
/// Requires the devkitARM toolchain for 3DS targets on the host system.

pub fn target() -> Target {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-specs=3dsx.specs".to_string(),
            "-mtune=mpcore".to_string(),
            "-mfloat-abi=hard".to_string(),
            "-mtp=soft".to_string(),
        ],
    );

    Target {
        llvm_target: "armv6k-none-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            os: "horizon".to_string(),
            env: "newlib".to_string(),
            vendor: "nintendo".to_string(),
            abi: "eabihf".to_string(),
            linker_flavor: LinkerFlavor::Gcc,
            cpu: "mpcore".to_string(),
            executables: true,
            families: vec!["unix".to_string()],
            linker: Some("arm-none-eabi-gcc".to_string()),
            relocation_model: RelocModel::Static,
            features: "+vfp2".to_string(),
            pre_link_args,
            exe_suffix: ".elf".to_string(),
            no_default_libraries: false,
            ..Default::default()
        },
    }
}
