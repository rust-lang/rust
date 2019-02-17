use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::apple_base::opts();
    base.cpu = "core2".to_string();
    base.max_atomic_width = Some(128); // core2 support cmpxchg16b
    base.eliminate_frame_pointer = false;
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m64".to_string()]);
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-apple-darwin".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "macos".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
