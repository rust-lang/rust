// This defines the amd64 target for the Linux Kernel. See the linux-kernel-base module for
// generic Linux kernel options.

use crate::spec::{CodeModel, LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::linux_kernel_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.features =
        "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,-avx,-avx2,+soft-float"
            .to_string();
    base.code_model = Some(CodeModel::Kernel);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m64".to_string());

    Target {
        // FIXME: Some dispute, the linux-on-clang folks think this should use "Linux"
        llvm_target: "x86_64-elf".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),

        options: base,
    }
}
