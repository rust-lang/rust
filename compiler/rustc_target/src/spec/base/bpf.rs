use rustc_abi::Endian;

use crate::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, TargetOptions};

pub(crate) fn opts(endian: Endian) -> TargetOptions {
    TargetOptions {
        allow_asm: true,
        endian,
        linker_flavor: LinkerFlavor::Bpf,
        atomic_cas: false,
        dynamic_linking: true,
        no_builtins: true,
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: true,
        // Disable MergeFunctions since:
        // - older kernels don't support bpf-to-bpf calls
        // - on newer kernels, userspace still needs to relocate before calling
        //   BPF_PROG_LOAD and not all BPF libraries do that yet
        merge_functions: MergeFunctions::Disabled,
        obj_is_bitcode: true,
        requires_lto: false,
        singlethread: true,
        // When targeting the `v3` cpu in llvm, 32-bit atomics are also supported.
        // But making this value change based on the target cpu can be mostly confusing
        // and would require a bit of a refactor.
        min_atomic_width: Some(64),
        max_atomic_width: Some(64),
        ..Default::default()
    }
}
