use rustc_abi::{Align, Endian};

use crate::spec::{SanitizerSet, StackProbeType, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.endian = Endian::Big;
    // z10 is the oldest CPU supported by LLVM
    base.cpu = "z10".into();
    base.max_atomic_width = Some(128);
    base.min_global_align = Some(Align::from_bits(16).unwrap());
    base.stack_probes = StackProbeType::Inline;
    base.supported_sanitizers =
        SanitizerSet::ADDRESS | SanitizerSet::LEAK | SanitizerSet::MEMORY | SanitizerSet::THREAD;

    Target {
        llvm_target: "s390x-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("S390x Linux (kernel 3.2, glibc 2.17)".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64".into(),
        arch: "s390x".into(),
        options: base,
    }
}
