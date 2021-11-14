use crate::abi::Endian;
use crate::spec::{LinkArgs, LinkerFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

/// A base target for PlayStation Vita devices using the VITASDK toolchain (using newlib).
///
/// Requires the VITASDK toolchain on the host system.

pub fn target() -> Target {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Gcc, vec!["-Wl,-q".to_string()]);

    Target {
        llvm_target: "armv7a-vita-newlibeabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            os: "vita".to_string(),
            endian: Endian::Little,
            c_int_width: "32".to_string(),
            dynamic_linking: false,
            env: "newlib".to_string(),
            vendor: "sony".to_string(),
            abi: "eabihf".to_string(),
            linker_flavor: LinkerFlavor::Gcc,
            linker_is_gnu: true,
            no_default_libraries: false,
            cpu: "cortex-a9".to_string(),
            executables: true,
            families: vec!["unix".to_string()],
            linker: Some("arm-vita-eabi-gcc".to_string()),
            relocation_model: RelocModel::Static,
            features: "+v7,+neon".to_string(),
            pre_link_args,
            exe_suffix: ".elf".to_string(),
            panic_strategy: PanicStrategy::Abort,
            max_atomic_width: Some(32),
            ..Default::default()
        },
    }
}
