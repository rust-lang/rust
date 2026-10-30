use std::collections::BTreeMap;

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
        // Atomic loads and stores up to 64 bits are supported from CPU v4.
        min_atomic_width: Some(8),
        max_atomic_width: Some(64),
        per_cpu_atomic_widths: BTreeMap::from([
            // CPU versions from v1 to v3 support only read-modify-write atomic
            // operations (atomicrmw, cmpxchg), they do not support atomic
            // loads and stores. Proof:
            // - src/llvm-project/llvm/test/CodeGen/BPF/atomic-load-store.ll
            //   works only on v4.
            // - All other atomic*.ll tests enabled on lower CPU versions don't
            //   perform loads and stores.
            // `Atomic*` types from `core::sync::atomic` require availability
            // of both kinds of operations, so they need to be disabled for
            // CPU versions lower than v4. BPF developers wanting to use r-m-w
            // operations on older kernels/CPUs can call intrinsics directly
            // instead of using `core::sync::atomic`.
            ("v1".into(), (8, 0)),
            ("v2".into(), (8, 0)),
            ("v3".into(), (8, 0)),
            ("generic".into(), (8, 0)), // generic == v3
        ]),
        ..Default::default()
    }
}
