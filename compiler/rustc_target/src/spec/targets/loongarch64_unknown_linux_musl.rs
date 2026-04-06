use crate::spec::{
    Arch, CodeModel, LlvmAbi, SanitizerSet, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "loongarch64-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some("LoongArch64 Linux (LP64D ABI) with musl 1.2.5".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: Arch::LoongArch64,
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic".into(),
            features: "+f,+d,+lsx,+relax".into(),
            llvm_abiname: LlvmAbi::Lp64d,
            max_atomic_width: Some(64),
            mcount: "_mcount".into(),
            crt_static_default: false,
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::LEAK
                | SanitizerSet::MEMORY
                | SanitizerSet::THREAD,
            supports_xray: true,
            direct_access_external_data: Some(false),
            ..base::linux_musl::opts()
        },
    }
}
