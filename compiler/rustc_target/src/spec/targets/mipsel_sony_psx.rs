use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions, cvs,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "mipsel-sony-psx".into(),
        metadata: TargetMetadata {
            description: Some("MIPS (LE) Sony PlayStation 1 (PSX)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),

        options: TargetOptions {
            // The Playstation 1 is mostly bare-metal, but the BIOS does provide some a slight bit
            // of functionality post load, so we still declare it as `cfg!(target_os = "psx")`.
            //
            // See <https://github.com/rust-lang/rust/pull/131168> for details.
            os: "psx".into(),
            vendor: "sony".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            cpu: "mips1".into(),
            executables: true,
            linker: Some("rust-lld".into()),
            relocation_model: RelocModel::Static,
            exe_suffix: ".exe".into(),

            // PSX doesn't natively support floats.
            features: "+soft-float".into(),

            // This should be 16 bits, but LLVM incorrectly tries emitting MIPS-II SYNC instructions
            // for atomic loads and stores. This crashes rustc so we have to disable the Atomic* API
            // until this is fixed upstream. See https://reviews.llvm.org/D122427#3420144 for more
            // info.
            max_atomic_width: Some(0),

            // PSX does not support trap-on-condition instructions.
            llvm_args: cvs!["-mno-check-zero-division"],
            llvm_abiname: "o32".into(),
            panic_strategy: PanicStrategy::Abort,
            ..Default::default()
        },
    }
}
