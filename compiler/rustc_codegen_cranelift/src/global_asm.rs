//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::ItemId;
use rustc_session::config::{OutputFilenames, OutputType};

use crate::prelude::*;

pub(crate) fn codegen_global_asm_item(tcx: TyCtxt<'_>, global_asm: &mut String, item_id: ItemId) {
    let item = tcx.hir().item(item_id);
    if let rustc_hir::ItemKind::GlobalAsm(asm) = item.kind {
        if !asm.options.contains(InlineAsmOptions::ATT_SYNTAX) {
            global_asm.push_str("\n.intel_syntax noprefix\n");
        } else {
            global_asm.push_str("\n.att_syntax\n");
        }
        for piece in asm.template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => global_asm.push_str(s),
                InlineAsmTemplatePiece::Placeholder { .. } => todo!(),
            }
        }
        global_asm.push_str("\n.att_syntax\n\n");
    } else {
        bug!("Expected GlobalAsm found {:?}", item);
    }
}

#[derive(Debug)]
pub(crate) struct GlobalAsmConfig {
    asm_enabled: bool,
    assembler: PathBuf,
    pub(crate) output_filenames: Arc<OutputFilenames>,
}

impl GlobalAsmConfig {
    pub(crate) fn new(tcx: TyCtxt<'_>) -> Self {
        let asm_enabled = cfg!(feature = "inline_asm") && !tcx.sess.target.is_like_windows;

        GlobalAsmConfig {
            asm_enabled,
            assembler: crate::toolchain::get_toolchain_binary(tcx.sess, "as"),
            output_filenames: tcx.output_filenames(()).clone(),
        }
    }
}

pub(crate) fn compile_global_asm(
    config: &GlobalAsmConfig,
    cgu_name: &str,
    global_asm: &str,
) -> Result<Option<PathBuf>, String> {
    if global_asm.is_empty() {
        return Ok(None);
    }

    if !config.asm_enabled {
        if global_asm.contains("__rust_probestack") {
            return Ok(None);
        }

        // FIXME fix linker error on macOS
        if cfg!(not(feature = "inline_asm")) {
            return Err(
                "asm! and global_asm! support is disabled while compiling rustc_codegen_cranelift"
                    .to_owned(),
            );
        } else {
            return Err("asm! and global_asm! are not yet supported on Windows".to_owned());
        }
    }

    // Remove all LLVM style comments
    let global_asm = global_asm
        .lines()
        .map(|line| if let Some(index) = line.find("//") { &line[0..index] } else { line })
        .collect::<Vec<_>>()
        .join("\n");

    let output_object_file = config.output_filenames.temp_path(OutputType::Object, Some(cgu_name));

    // Assemble `global_asm`
    let global_asm_object_file = add_file_stem_postfix(output_object_file.clone(), ".asm");
    let mut child = Command::new(&config.assembler)
        .arg("-o")
        .arg(&global_asm_object_file)
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to spawn `as`.");
    child.stdin.take().unwrap().write_all(global_asm.as_bytes()).unwrap();
    let status = child.wait().expect("Failed to wait for `as`.");
    if !status.success() {
        return Err(format!("Failed to assemble `{}`", global_asm));
    }

    Ok(Some(global_asm_object_file))
}

pub(crate) fn add_file_stem_postfix(mut path: PathBuf, postfix: &str) -> PathBuf {
    let mut new_filename = path.file_stem().unwrap().to_owned();
    new_filename.push(postfix);
    if let Some(extension) = path.extension() {
        new_filename.push(".");
        new_filename.push(extension);
    }
    path.set_file_name(new_filename);
    path
}
