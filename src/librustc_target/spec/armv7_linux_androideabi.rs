use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

// This target if is for the baseline of the Android v7a ABI
// in thumb mode. It's named armv7-* instead of thumbv7-*
// for historical reasons. See the thumbv7neon variant for
// enabling NEON.

// See https://developer.android.com/ndk/guides/abis.html#v7a
// for target ABI requirements.

pub fn target() -> TargetResult {
    let mut base = super::android_base::opts();
    base.features = "+v7,+thumb-mode,+thumb2,+vfp3,+d16,-neon".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args
        .get_mut(&LinkerFlavor::Gcc).unwrap().push("-march=armv7-a".to_string());

    Ok(Target {
        llvm_target: "armv7-none-linux-android".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "android".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. base
        },
    })
}
