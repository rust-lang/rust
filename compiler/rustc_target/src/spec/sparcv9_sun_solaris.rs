use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::solaris_base::opts();
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m64".to_string()]);
    // llvm calls this "v9"
    base.cpu = "v9".to_string();
    base.max_atomic_width = Some(64);

    Target {
        llvm_target: "sparcv9-sun-solaris".to_string(),
        target_endian: "big".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i64:64-n32:64-S128".to_string(),
        // Use "sparc64" instead of "sparcv9" here, since the former is already
        // used widely in the source base.  If we ever needed ABI
        // differentiation from the sparc64, we could, but that would probably
        // just be confusing.
        arch: "sparc64".to_string(),
        target_os: "solaris".to_string(),
        target_env: String::new(),
        target_vendor: "sun".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    }
}
