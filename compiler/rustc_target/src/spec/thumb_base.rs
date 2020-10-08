// These `thumbv*` targets cover the ARM Cortex-M family of processors which are widely used in
// microcontrollers. Namely, all these processors:
//
// - Cortex-M0
// - Cortex-M0+
// - Cortex-M1
// - Cortex-M3
// - Cortex-M4(F)
// - Cortex-M7(F)
// - Cortex-M23
// - Cortex-M33
//
// We have opted for these instead of one target per processor (e.g., `cortex-m0`, `cortex-m3`,
// etc) because the differences between some processors like the cortex-m0 and cortex-m1 are almost
// non-existent from the POV of codegen so it doesn't make sense to have separate targets for them.
// And if differences exist between two processors under the same target, rustc flags can be used to
// optimize for one processor or the other.
//
// Also, we have not chosen a single target (`arm-none-eabi`) like GCC does because this makes
// difficult to integrate Rust code and C code. Targeting the Cortex-M4 requires different gcc flags
// than the ones you would use for the Cortex-M0 and with a single target it'd be impossible to
// differentiate one processor from the other.
//
// About arm vs thumb in the name. The Cortex-M devices only support the Thumb instruction set,
// which is more compact (higher code density), and not the ARM instruction set. That's why LLVM
// triples use thumb instead of arm. We follow suit because having thumb in the name let us
// differentiate these targets from our other `arm(v7)-*-*-gnueabi(hf)` targets in the context of
// build scripts / gcc flags.

use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, TargetOptions};

pub fn opts() -> TargetOptions {
    // See rust-lang/rfcs#1645 for a discussion about these defaults
    TargetOptions {
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        executables: true,
        // In most cases, LLD is good enough
        linker: Some("rust-lld".to_string()),
        // Because these devices have very little resources having an unwinder is too onerous so we
        // default to "abort" because the "unwind" strategy is very rare.
        panic_strategy: PanicStrategy::Abort,
        // Similarly, one almost always never wants to use relocatable code because of the extra
        // costs it involves.
        relocation_model: RelocModel::Static,
        unsupported_abis: super::arm_base::unsupported_abis(),
        // When this section is added a volatile load to its start address is also generated. This
        // volatile load is a footgun as it can end up loading an invalid memory address, depending
        // on how the user set up their linker scripts. This section adds pretty printer for stuff
        // like std::Vec, which is not that used in no-std context, so it's best to left it out
        // until we figure a way to add the pretty printers without requiring a volatile load cf.
        // rust-lang/rust#44993.
        emit_debug_gdb_scripts: false,
        // LLVM is eager to trash the link register when calling `noreturn` functions, which
        // breaks debugging. Preserve LR by default to prevent that from happening.
        eliminate_frame_pointer: false,
        ..Default::default()
    }
}
