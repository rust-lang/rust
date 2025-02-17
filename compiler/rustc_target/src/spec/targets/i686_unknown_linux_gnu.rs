use crate::spec::{
    Cc, LinkerFlavor, Lld, RustcAbi, SanitizerSet, StackProbeType, Target, TargetMetadata, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.rustc_abi = Some(RustcAbi::X86Sse2);
    // Dear distribution packager, if you are changing the base CPU model with the goal of removing
    // the SSE2 requirement, make sure to also set the `rustc_abi` to `None` above or else the compiler
    // will complain that the chosen ABI cannot be realized with the given CPU features.
    // Also note that x86 without SSE2 is *not* considered a Tier 1 target by the Rust project, and
    // it has some known floating-point correctness issues mostly caused by a lack of people caring
    // for LLVM's x87 support (double-rounding, value truncation; see
    // <https://github.com/rust-lang/rust/issues/114479> for details). This can lead to incorrect
    // math (Rust generally promises exact math, so this can break code in unexpected ways) and it
    // can lead to memory safety violations if floating-point values are used e.g. to access an
    // array. If users run into such issues and report bugs upstream and then it turns out that the
    // bugs are caused by distribution patches, that leads to confusion and frustration.
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.supported_sanitizers = SanitizerSet::ADDRESS;
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-m32"]);
    base.stack_probes = StackProbeType::Inline;

    Target {
        llvm_target: "i686-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("32-bit Linux (kernel 3.2, glibc 2.17+)".into()),
            tier: Some(1),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
