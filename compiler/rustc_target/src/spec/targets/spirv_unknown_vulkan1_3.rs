use crate::spec::{Arch, LinkerFlavor, Os, PanicStrategy, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "spirv-unknown-vulkan1.3".into(),
        metadata: TargetMetadata {
            description: Some("Vulkan 1.3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10".into(),
        arch: Arch::SpirV,
        options: TargetOptions {
            os: Os::Vulkan,
            vendor: "unknown".into(),
            linker_flavor: LinkerFlavor::Llbc,
            max_atomic_width: Some(32),
            panic_strategy: PanicStrategy::Abort,
            // Allow `cdylib` crate type.
            dynamic_linking: true,
            obj_is_bitcode: true,
            only_cdylib: true,
            dll_prefix: "".into(),
            dll_suffix: ".spvt".into(),
            is_like_gpu: true,
            // The LLVM backend does not support stack canaries for this target
            supports_stack_protector: false,

            // Static initializers must not have cycles on this target
            static_initializer_must_be_acyclic: true,
            ..Default::default()
        },
    }
}
