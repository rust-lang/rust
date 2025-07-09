use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        arch: "amdgpu".into(),
        data_layout: "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9".into(),
        llvm_target: "amdgcn-amd-amdhsa".into(),
        metadata: TargetMetadata {
            description: Some("AMD GPU".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,

        options: TargetOptions {
            os: "amdhsa".into(),
            vendor: "amd".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),

            // There are many CPUs, one for each hardware generation.
            // Require to set one explicitly as there is no good default.
            need_explicit_cpu: true,

            max_atomic_width: Some(64),

            // Unwinding on GPUs is not useful.
            panic_strategy: PanicStrategy::Abort,

            // amdgpu backend does not support libcalls.
            no_builtins: true,
            simd_types_indirect: false,

            // Allow `cdylib` crate type.
            dynamic_linking: true,
            only_cdylib: true,
            executables: false,
            dll_prefix: "".into(),
            dll_suffix: ".elf".into(),

            // The LLVM backend does not support stack canaries for this target
            supports_stack_protector: false,

            // Force LTO, object linking does not yet work with amdgpu.
            requires_lto: true,

            ..Default::default()
        },
    }
}
