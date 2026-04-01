// These are the baseline settings for 32-bit bare-metal Arm targets using the EABI or EABIHF ABI.

use crate::spec::{Cc, FramePointer, LinkerFlavor, Lld, PanicStrategy, RelocModel, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    // See rust-lang/rfcs#1645 for a discussion about these defaults
    TargetOptions {
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        // In most cases, LLD is good enough
        linker: Some("rust-lld".into()),
        // Because these devices have very little resources having an unwinder is too onerous so we
        // default to "abort" because the "unwind" strategy is very rare.
        panic_strategy: PanicStrategy::Abort,
        // Similarly, one almost always never wants to use relocatable code because of the extra
        // costs it involves.
        relocation_model: RelocModel::Static,
        // When this section is added a volatile load to its start address is also generated. This
        // volatile load is a footgun as it can end up loading an invalid memory address, depending
        // on how the user set up their linker scripts. This section adds pretty printer for stuff
        // like std::Vec, which is not that used in no-std context, so it's best to left it out
        // until we figure a way to add the pretty printers without requiring a volatile load cf.
        // rust-lang/rust#44993.
        emit_debug_gdb_scripts: false,
        // LLVM is eager to trash the link register when calling `noreturn` functions, which
        // breaks debugging. Preserve LR by default to prevent that from happening.
        frame_pointer: FramePointer::Always,
        // ARM supports multiple ABIs for enums, the linux one matches the default of 32 here
        // but any arm-none or thumb-none target will be defaulted to 8 on GCC.
        c_enum_min_bits: Some(8),
        ..Default::default()
    }
}
