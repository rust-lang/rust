use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetMetadata, TargetOptions, base};

// This target is for glibc Linux on Csky

pub(crate) fn target() -> Target {
    Target {
        //https://github.com/llvm/llvm-project/blob/8b76aea8d8b1b71f6220bc2845abc749f18a19b7/clang/lib/Basic/Targets/CSKY.h
        llvm_target: "csky-unknown-linux-gnuabiv2".into(),
        metadata: TargetMetadata {
            description: Some("C-SKY abiv2 Linux, hardfloat (little endian)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true)
        },
        pointer_width: 32,
        data_layout: "e-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32-v128:32:32-a:0:32-Fi32-n32".into(),
        arch: "csky".into(),
        options: TargetOptions {
            abi: "abiv2hf".into(),
            cpu: "ck860fv".into(),
            features: "+hard-float,+hard-float-abi,+2e3,+3e7,+7e10,+cache,+dsp1e2,+dspe60,+e1,+e2,+edsp,+elrw,+hard-tp,+high-registers,+hwdiv,+mp,+mp1e2,+nvic,+trust".into(),
            late_link_args: TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-l:libatomic.a", "-mhard-float"]),
            max_atomic_width: Some(32),
            ..base::linux_gnu::opts()
        },
    }
}
