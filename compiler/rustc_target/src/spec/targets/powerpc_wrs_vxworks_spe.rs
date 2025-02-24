use rustc_abi::Endian;

use crate::spec::{
    Cc, LinkerFlavor, Lld, StackProbeType, Target, TargetMetadata, TargetOptions, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::vxworks::opts();
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-mspe", "--secure-plt"]);
    base.max_atomic_width = Some(32);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "powerpc-unknown-linux-gnuspe".into(),
        metadata: TargetMetadata {
            description: None,
            tier: Some(3),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:32-Fn32-i64:64-n32".into(),
        arch: "powerpc".into(),
        options: TargetOptions {
            abi: "spe".into(),
            endian: Endian::Big,
            // feature msync would disable instruction 'fsync' which is not supported by fsl_p1p2
            features: "+secure-plt,+msync".into(),
            ..base
        },
    }
}
