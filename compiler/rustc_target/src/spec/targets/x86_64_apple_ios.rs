use crate::spec::base::apple::{ios_sim_llvm_target, opts, Arch};
use crate::spec::{SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    // x86_64-apple-ios is a simulator target, even though it isn't declared
    // that way in the target name like the other ones...
    let arch = Arch::X86_64_sim;
    let mut base = opts("ios", arch);
    base.supported_sanitizers = SanitizerSet::ADDRESS | SanitizerSet::THREAD;

    Target {
        llvm_target: ios_sim_llvm_target(arch).into(),
        description: None,
        pointer_width: 64,
        data_layout:
            "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions { max_atomic_width: Some(128), ..base },
    }
}
