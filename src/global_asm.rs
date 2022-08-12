//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::ItemId;
use rustc_session::config::OutputType;

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

pub(crate) fn compile_global_asm(
    tcx: TyCtxt<'_>,
    cgu_name: &str,
    global_asm: &str,
) -> io::Result<()> {
    if global_asm.is_empty() {
        return Ok(());
    }

    if cfg!(not(feature = "inline_asm"))
        || tcx.sess.target.is_like_osx
        || tcx.sess.target.is_like_windows
    {
        if global_asm.contains("__rust_probestack") {
            return Ok(());
        }

        // FIXME fix linker error on macOS
        if cfg!(not(feature = "inline_asm")) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "asm! and global_asm! support is disabled while compiling rustc_codegen_cranelift",
            ));
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "asm! and global_asm! are not yet supported on macOS and Windows",
            ));
        }
    }

    let assembler = crate::toolchain::get_toolchain_binary(tcx.sess, "as");
    let linker = crate::toolchain::get_toolchain_binary(tcx.sess, "ld");

    // Remove all LLVM style comments
    let global_asm = global_asm
        .lines()
        .map(|line| if let Some(index) = line.find("//") { &line[0..index] } else { line })
        .collect::<Vec<_>>()
        .join("\n");

    let output_object_file = tcx.output_filenames(()).temp_path(OutputType::Object, Some(cgu_name));

    // Assemble `global_asm`
    let global_asm_object_file = add_file_stem_postfix(output_object_file.clone(), ".asm");
    let mut child = Command::new(assembler)
        .arg("-o")
        .arg(&global_asm_object_file)
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to spawn `as`.");
    child.stdin.take().unwrap().write_all(global_asm.as_bytes()).unwrap();
    let status = child.wait().expect("Failed to wait for `as`.");
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to assemble `{}`", global_asm),
        ));
    }

    // Link the global asm and main object file together
    let main_object_file = add_file_stem_postfix(output_object_file.clone(), ".main");
    std::fs::rename(&output_object_file, &main_object_file).unwrap();
    let status = Command::new(linker)
        .arg("-r") // Create a new object file
        .arg("-o")
        .arg(output_object_file)
        .arg(&main_object_file)
        .arg(&global_asm_object_file)
        .status()
        .unwrap();
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Failed to link `{}` and `{}` together",
                main_object_file.display(),
                global_asm_object_file.display(),
            ),
        ));
    }

    std::fs::remove_file(global_asm_object_file).unwrap();
    std::fs::remove_file(main_object_file).unwrap();

    Ok(())
}

fn add_file_stem_postfix(mut path: PathBuf, postfix: &str) -> PathBuf {
    let mut new_filename = path.file_stem().unwrap().to_owned();
    new_filename.push(postfix);
    if let Some(extension) = path.extension() {
        new_filename.push(".");
        new_filename.push(extension);
    }
    path.set_file_name(new_filename);
    path
}
