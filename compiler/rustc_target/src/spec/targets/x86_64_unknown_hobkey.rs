use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, Target, TargetMetadata, TargetOptions};
use crate::spec::base;
pub(crate) fn target() -> Target{
    let mut base : TargetOptions = base::hobkey::opts();
    base.cpu = "x86-64".into();
    base.linker_flavor = LinkerFlavor::Gnu(Cc::No, Lld::Yes);
    base.linker = Some("rust-lld".into());
    base.panic_strategy = PanicStrategy::Abort;

    Target{ 
        llvm_target: "x86_64-unknown-none-elf".into(),
        metadata: TargetMetadata{
            description: None,
            tier: None,
            host_tools: None,
            std: Some(true)
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base
        
    }
}