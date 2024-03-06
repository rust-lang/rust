use crate::spec::base::apple::{ios_sim_llvm_target, opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    // i386-apple-ios is a simulator target, even though it isn't declared
    // that way in the target name like the other ones...
    let arch = Arch::I386_sim;
    Target {
        // Clang automatically chooses a more specific target based on
        // IPHONEOS_DEPLOYMENT_TARGET.
        // This is required for the target to pick the right
        // MACH-O commands, so we do too.
        llvm_target: ios_sim_llvm_target(arch).into(),
        description: None,
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
            .into(),
        arch: arch.target_arch(),
        options: TargetOptions { max_atomic_width: Some(64), ..opts("ios", arch) },
    }
}
