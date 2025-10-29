//! Targets the ARMv5TE, with code as `a32` code by default.

use crate::spec::{
    Abi, Arch, FloatAbi, FramePointer, Target, TargetMetadata, TargetOptions, base, cvs,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv5te-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv5TE".into()),
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
            // extra args passed to the external assembler (assuming `arm-none-eabi-as`):
            // * activate t32/a32 interworking
            // * use arch ARMv5TE
            // * use little-endian
            asm_args: cvs!["-mthumb-interwork", "-march=armv5te", "-mlittle-endian",],
            // minimum extra features, these cannot be disabled via -C
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
