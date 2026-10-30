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
        singlethread: true,
        // CPU v4 provides `load-acq-store-rel` BPF subtarget that is needed
        // for the 64-bit atomic load/store. It is required to build entire
        // core library to BPF.
        cpu: "v4".into(),
        min_atomic_width: Some(64),
        max_atomic_width: Some(64),
        ..Default::default()
    }
}
