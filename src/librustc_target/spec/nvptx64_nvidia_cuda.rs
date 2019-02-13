use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult, PanicStrategy, MergeFunctions};
use crate::spec::abi::Abi;

pub fn target() -> TargetResult {
    Ok(Target {
        arch: "nvptx64".to_string(),
        data_layout: "e-i64:64-i128:128-v16:16-v32:32-n16:32:64".to_string(),
        llvm_target: "nvptx64-nvidia-cuda".to_string(),

        target_os: "cuda".to_string(),
        target_vendor: "nvidia".to_string(),
        target_env: String::new(),

        linker_flavor: LinkerFlavor::PtxLinker,

        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),

        options: TargetOptions {
            // The linker can be installed from `crates.io`.
            linker: Some("rust-ptx-linker".to_string()),

            // With `ptx-linker` approach, it can be later overriden via link flags.
            cpu: "sm_30".to_string(),

            // FIXME: create tests for the atomics.
            max_atomic_width: Some(64),

            // Unwinding on CUDA is neither feasible nor useful.
            panic_strategy: PanicStrategy::Abort,

            // Needed to use `dylib` and `bin` crate types and the linker.
            dynamic_linking: true,
            executables: true,

            // Avoid using dylib because it contain metadata not supported
            // by LLVM NVPTX backend.
            only_cdylib: true,

            // Let the `ptx-linker` to handle LLVM lowering into MC / assembly.
            obj_is_bitcode: true,

            // Convinient and predicable naming scheme.
            dll_prefix: "".to_string(),
            dll_suffix: ".ptx".to_string(),
            exe_suffix: ".ptx".to_string(),

            // Disable MergeFunctions LLVM optimisation pass because it can
            // produce kernel functions that call other kernel functions.
            // This behavior is not supported by PTX ISA.
            merge_functions: MergeFunctions::Disabled,

            // FIXME: enable compilation tests for the target and
            // create the tests for this.
            abi_blacklist: vec![
                Abi::Cdecl,
                Abi::Stdcall,
                Abi::Fastcall,
                Abi::Vectorcall,
                Abi::Thiscall,
                Abi::Aapcs,
                Abi::Win64,
                Abi::SysV64,
                Abi::Msp430Interrupt,
                Abi::X86Interrupt,
                Abi::AmdGpuKernel,
            ],

            .. Default::default()
        },
    })
}
