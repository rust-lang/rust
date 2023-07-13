use crate::spec::{Target, TargetOptions};

// This target is for glibc Linux on Csky
// hardfloat.

pub fn target() -> Target {
    Target {
        //https://github.com/llvm/llvm-project/blob/8b76aea8d8b1b71f6220bc2845abc749f18a19b7/clang/lib/Basic/Targets/CSKY.h
        llvm_target: "csky-unknown-linux".into(),
        pointer_width: 32,
        data_layout: "e-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32-v128:32:32-a:0:32-Fi32-n32".into(),
        arch: "csky".into(),
        options: TargetOptions {
            abi: "abiv2".into(),
            //+hard-float, +hard-float-abi, +fpuv2_sf, +fpuv2_df, +fpuv3_sf, +fpuv3_df,  +vdspv2, +dspv2, +vdspv1, +3e3r1
            features: "".into(),
            max_atomic_width: Some(32),
            // mcount: "\u{1}__gnu_mcount_nc".into(),
            ..super::linux_gnu_base::opts()
        },
    }
}
