use crate::abi::Endian;
use crate::spec::{cvs, Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

/// A base target for PlayStation Vita devices using the VITASDK toolchain (using newlib).
///
/// Requires the VITASDK toolchain on the host system.

pub fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-Wl,-q"]);

    Target {
        llvm_target: "armv7a-vita-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            os: "vita".into(),
            endian: Endian::Little,
            c_int_width: "32".into(),
            dynamic_linking: false,
            env: "newlib".into(),
            vendor: "sony".into(),
            abi: "eabihf".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            no_default_libraries: false,
            cpu: "cortex-a9".into(),
            executables: true,
            families: cvs!["unix"],
            linker: Some("arm-vita-eabi-gcc".into()),
            relocation_model: RelocModel::Static,
            features: "+v7,+neon".into(),
            pre_link_args,
            exe_suffix: ".elf".into(),
            panic_strategy: PanicStrategy::Abort,
            max_atomic_width: Some(64),
            ..Default::default()
        },
    }
}
