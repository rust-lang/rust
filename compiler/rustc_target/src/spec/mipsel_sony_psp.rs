use crate::spec::{cvs, Target, TargetOptions};
use crate::spec::{LinkerFlavor, LldFlavor, RelocModel};

// The PSP has custom linker requirements.
const LINKER_SCRIPT: &str = include_str!("./mipsel_sony_psp_linker_script.ld");

pub fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(LinkerFlavor::Ld, &["--emit-relocs", "--nmagic"]);

    Target {
        llvm_target: "mipsel-sony-psp".into(),
        pointer_width: 32,
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),
        arch: "mips".into(),

        options: TargetOptions {
            os: "psp".into(),
            vendor: "sony".into(),
            linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
            cpu: "mips2".into(),
            executables: true,
            linker: Some("rust-lld".into()),
            relocation_model: RelocModel::Static,

            // PSP FPU only supports single precision floats.
            features: "+single-float".into(),

            // PSP does not support trap-on-condition instructions.
            llvm_args: cvs!["-mno-check-zero-division"],
            pre_link_args,
            link_script: Some(LINKER_SCRIPT.into()),
            ..Default::default()
        },
    }
}
