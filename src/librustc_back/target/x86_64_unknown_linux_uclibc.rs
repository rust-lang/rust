use target::Target;

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = 64;
    base.pre_link_args.push("-m64".to_string());
    Target {
        llvm_target: "x86_64-unknown-linux-uclibc".to_string(),
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        arch: "x86_64".to_string(),
        target_os: "linux".to_string(),
        target_env: "uclibc".to_string(),
        target_vendor: "unknown".to_string(),
        options: base,
    }
}
