use crate::spec::{
	Arch, Target, TargetMetadata, base
};

pub(crate) fn target() -> Target {
    // Get the base UEFI configuration
    let mut base = base::uefi_msvc::opts();
    
    // Override with RISC-V specific settings
    base.cpu = "generic-rv64".into();
    base.features = "+m,+a,+f,+d,+c".into();
    base.max_atomic_width = Some(64);
    base.atomic_cas = true;
    base.disable_redzone = true;

    Target {
        llvm_target: "riscv64".into(),
        metadata: TargetMetadata {
            description: Some("Bare RISC-V (RV64IMAFDC ISA) UEFI".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::RiscV64,
        
        options: base,
    }
}
