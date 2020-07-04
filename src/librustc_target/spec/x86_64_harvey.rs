// This defines the x86_64 target for the Harvey Kernel. See the harvey-kernel-base module for
// generic Harvey kernel options. Note that in Harvey, as in Plan 9 and Go, we call x86_64
// amd64.

use crate::spec::{CodeModel, LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::harvey_kernel_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.features =
        "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,-avx,-avx2,+soft-float"
            .to_string();
    base.code_model = Some(CodeModel::Kernel);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());

    Ok(Target {
        llvm_target: "x86_64-elf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        target_os: "none".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        arch: "x86_64".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: base,
    })
}
