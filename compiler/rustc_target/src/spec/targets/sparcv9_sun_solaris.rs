use crate::abi::Endian;
use crate::spec::{base, Cc, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::solaris::opts();
    base.endian = Endian::Big;
    base.pre_link_args = TargetOptions::link_args(LinkerFlavor::Unix(Cc::Yes), &["-m64"]);
    // llvm calls this "v9"
    base.cpu = "v9".into();
    base.vendor = "sun".into();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparcv9-sun-solaris".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".into(),
        // Use "sparc64" instead of "sparcv9" here, since the former is already
        // used widely in the source base. If we ever needed ABI
        // differentiation from the sparc64, we could, but that would probably
        // just be confusing.
        arch: "sparc64".into(),
        options: base,
    }
}
