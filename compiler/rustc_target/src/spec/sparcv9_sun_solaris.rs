use crate::abi::Endian;
use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::solaris_base::opts();
    base.endian = Endian::Big;
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m64".to_string()]);
    // llvm calls this "v9"
    base.cpu = "v9".to_string();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparcv9-sun-solaris".to_string(),
        pointer_width: 64,
        data_layout: "E-m:e-i64:64-n32:64-S128".to_string(),
        // Use "sparc64" instead of "sparcv9" here, since the former is already
        // used widely in the source base.  If we ever needed ABI
        // differentiation from the sparc64, we could, but that would probably
        // just be confusing.
        arch: "sparc64".to_string(),
        options: base,
    }
}
