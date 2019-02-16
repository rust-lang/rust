use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::android_base::opts();
    base.cpu = "x86-64".to_string();
    // https://developer.android.com/ndk/guides/abis.html#86-64
    base.features = "+mmx,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "x86_64-linux-android".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        target_os: "android".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
