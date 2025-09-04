use crate::spec::{
    LinkSelfContainedDefault, LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        arch: "nvptx64".into(),
        data_layout: "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64".into(),
        llvm_target: "nvptx64-nvidia-cuda".into(),
        metadata: TargetMetadata {
            description: Some("--emit=asm generates PTX code that runs on NVIDIA GPUs".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,

        options: TargetOptions {
            os: "cuda".into(),
            vendor: "nvidia".into(),
            linker_flavor: LinkerFlavor::Ptx,
            // The linker can be installed from `crates.io`.
            linker: Some("rust-ptx-linker".into()),

            // With `ptx-linker` approach, it can be later overridden via link flags.
            cpu: "sm_30".into(),

            // FIXME: create tests for the atomics.
            max_atomic_width: Some(64),

            // Unwinding on CUDA is neither feasible nor useful.
            panic_strategy: PanicStrategy::Abort,

            // Needed to use `dylib` and `bin` crate types and the linker.
            dynamic_linking: true,

            // Avoid using dylib because it contain metadata not supported
            // by LLVM NVPTX backend.
            only_cdylib: true,

            // Let the `ptx-linker` to handle LLVM lowering into MC / assembly.
            obj_is_bitcode: true,

            // Convenient and predicable naming scheme.
            dll_prefix: "".into(),
            dll_suffix: ".ptx".into(),
            exe_suffix: ".ptx".into(),

            // Disable MergeFunctions LLVM optimisation pass because it can
            // produce kernel functions that call other kernel functions.
            // This behavior is not supported by PTX ISA.
            merge_functions: MergeFunctions::Disabled,

            // The LLVM backend does not support stack canaries for this target
            supports_stack_protector: false,

            // Support using `self-contained` linkers like the llvm-bitcode-linker
            link_self_contained: LinkSelfContainedDefault::True,

            ..Default::default()
        },
    }
}
