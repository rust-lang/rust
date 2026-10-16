use crate::spec::{Target, TargetMetadata, TargetOptions};

// The Vega OS armv7 userspace, including the SDK sysroot, is built for Cortex-A7
// with NEON and VFPv4, so unlike armv7-unknown-linux-gnueabihf this target
// assumes both.

pub(crate) fn target() -> Target {
    let mut base = super::armv7_unknown_linux_gnueabihf::target();

    base.metadata = TargetMetadata {
        description: Some("Armv7-A Vega OS, hardfloat, Cortex-A7 with NEON (glibc 2.35)".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };

    base.options = TargetOptions {
        vendor: "vega".into(),
        cpu: "cortex-a7".into(),
        features: "+v7,+db,+dsp,+aclass,+perfmon,+vfp4,+neon,+thumb2".into(),
        ..base.options
    };

    base
}
