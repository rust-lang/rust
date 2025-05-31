use crate::spec::{Cc, FloatAbi, LinkerFlavor, Lld, Target, base};

pub(crate) fn target() -> Target {
    let mut base = base::helenos::opts();
    base.abi = "eabi".into();
    base.llvm_floatabi = Some(FloatAbi::Soft);
    base.max_atomic_width = Some(32);
    base.features = "+soft-float,+strict-align,+atomics-32".into();
    base.has_thumb_interworking = true;
    base.linker = Some("arm-helenos-gcc".into());
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-march=armv5te"]);

    // FIXME: these 3 flags are a hack to avoid generating R_*_RELATIVE relocations in code segment,
    // which cause the HelenOS loader to segfault. I believe the underlying issue is that HelenOS
    // doesn't map the code segment as writable, so the loader can't apply the relocations.
    // The same issue was with the i686-helenos target, I don't recall why the current combination
    // of flags avoids the issue there.
    base.crt_static_default = true;
    base.crt_static_respected = false;
    base.crt_static_allows_dylibs = true;

    Target {
        llvm_target: "armv5te-unknown-helenos-eabi".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("ARMv5te HelenOS".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: base,
    }
}
