use crate::spec::{
	Arch, CodeModel, LinkerFlavor, Lld, PanicStrategy, RelocModel,
	Target, TargetOptions, TargetMetadata, Os
};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        metadata: TargetMetadata {
            description: Some("Bare RISC-V (RV64IMAFDC ISA) UEFI".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        llvm_target: "riscv64gc-unknown-windows".into(),
        pointer_width: 64,
        arch: Arch::RiscV64,
        
        options: TargetOptions {
            os: Os::Uefi,
            vendor: "unknown".into(),
            linker_flavor: LinkerFlavor::Msvc(Lld::No),
            
            // UEFI characteristics
            executables: true,
            is_like_windows: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Pic,
            
            // RISC-V features
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c".into(),
            
            // These are the current correct field names:
            is_like_aix: false,
            is_like_android: false,
            is_like_msvc: true,
            
            // Codegen options
            code_model: Some(CodeModel::Medium),
            disable_redzone: true,
            ..Default::default()
        },
    }
}
