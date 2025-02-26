use rustc_abi::Endian;

use crate::spec::{CodeModel, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    let options = TargetOptions {
        cpu: "M68010".into(),
        max_atomic_width: None,
        endian: Endian::Big,
        // LLD currently does not have support for M68k
        linker: Some("m68k-linux-gnu-ld".into()),
        panic_strategy: PanicStrategy::Abort,
        code_model: Some(CodeModel::Medium),
        has_rpath: false,
        // should be soft-float
        llvm_floatabi: None,
        relocation_model: RelocModel::Static,
        ..Default::default()
    };

    Target {
        llvm_target: "m68k".into(),
        metadata: TargetMetadata {
            description: Some("Motorola 680x0".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:16:32-i8:8:8-i16:16:16-i32:16:32-n8:16:32-a:0:16-S16".into(),
        arch: "m68k".into(),
        options,
    }
}
