//! Targets the ARMv4T, with code as `a32` code by default.
//!
//! Primarily of use for the GBA, but usable with other devices too.
//!
//! Please ping @Lokathor if changes are needed.
//!
//! **Important:** This target profile **does not** specify a linker script. You
//! just get the default link script when you build a binary for this target.
//! The default link script is very likely wrong, so you should use
//! `-Clink-arg=-Tmy_script.ld` to override that with a correct linker script.

use crate::spec::{
    Abi, Arch, Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions, cvs,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv4t-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv4T".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: Arch::Arm,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        options: TargetOptions {
            abi: Abi::Eabi,
            llvm_floatabi: Some(FloatAbi::Soft),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            asm_args: cvs!["-mthumb-interwork", "-march=armv4t", "-mlittle-endian",],
            // Force-enable 32-bit atomics, which allows the use of atomic load/store only.
            // The resulting atomics are ABI incompatible with atomics backed by libatomic.
            features: "+soft-float,+strict-align".into(),
            atomic_cas: false,
            has_thumb_interworking: true,
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
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
        },
    }
}
