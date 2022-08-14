use crate::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        arch: "nvptx64".into(),
        data_layout: "e-i64:64-i128:128-v16:16-v32:32-n16:32:64".into(),
        llvm_target: "nvptx64-nvidia-cuda".into(),
        pointer_width: 64,

        options: TargetOptions {
            os: "cuda".into(),
            vendor: "nvidia".into(),
            linker_flavor: LinkerFlavor::Ptx,
            // The linker can be installed from `crates.io`.
            linker: Some("rust-ptx-linker".into()),
            linker_is_gnu: false,

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

            ..Default::default()
        },
    }
}
