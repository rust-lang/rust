use crate::spec::{LinkerFlavor, Target, TargetResult};

// See https://developer.android.com/ndk/guides/abis.html#x86
// for target ABI requirements.

pub fn target() -> TargetResult {
    let mut base = super::android_base::opts();

    base.max_atomic_width = Some(64);

    // http://developer.android.com/ndk/guides/abis.html#x86
    base.cpu = "pentiumpro".to_string();
    base.features = "+mmx,+sse,+sse2,+sse3,+ssse3".to_string();
    base.stack_probes = true;

    Ok(Target {
        llvm_target: "i686-linux-android".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "android".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
