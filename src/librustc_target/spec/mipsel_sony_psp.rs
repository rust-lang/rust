use std::{io, fs, env, path::PathBuf};
use crate::spec::{LinkerFlavor, LldFlavor, LinkArgs, RelocModel};
use crate::spec::{Target, TargetOptions, TargetResult};

// The PSP has custom linker requirements.
const LINKER_SCRIPT: &str = include_str!("./mipsel_sony_psp_linker_script.ld");

fn write_script() -> io::Result<PathBuf> {
    let path = env::temp_dir().join("rustc-mipsel-sony-psp-linkfile.ld");
    fs::write(&path, LINKER_SCRIPT)?;
    Ok(path)
}

pub fn target() -> TargetResult {
    let script = write_script().map_err(|e| {
        format!("failed to write link script: {}", e)
    })?;

    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld),
        vec![
            "--eh-frame-hdr".to_string(),
            "--emit-relocs".to_string(),
            "--script".to_string(),
            script.display().to_string(),
        ],
    );

    Ok(Target {
        llvm_target: "mipsel-sony-psp".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".to_string(),
        arch: "mips".to_string(),
        target_os: "psp".to_string(),
        target_env: "".to_string(),
        target_vendor: "sony".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            cpu: "mips2".to_string(),
            executables: true,
            linker: Some("rust-lld".to_owned()),
            relocation_model: RelocModel::Static,

            // PSP FPU only supports single precision floats.
            features: "+single-float".to_string(),

            // PSP does not support trap-on-condition instructions.
            llvm_args: vec![
                "-mno-check-zero-division".to_string(),
            ],
            pre_link_args,
            ..Default::default()
        },
    })
}
