use crate::spec::Target;

// See https://developer.android.com/ndk/guides/abis.html#x86
// for target ABI requirements.

pub fn target() -> Target {
    let mut base = super::android_base::opts();

    base.max_atomic_width = Some(64);

    // http://developer.android.com/ndk/guides/abis.html#x86
    base.cpu = "pentiumpro".to_string();
    base.features = "+mmx,+sse,+sse2,+sse3,+ssse3".to_string();
    base.stack_probes = true;

    Target {
        llvm_target: "i686-linux-android".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:32-n8:16:32-S128"
            .to_string(),
        arch: "x86".to_string(),
        options: base,
    }
}
