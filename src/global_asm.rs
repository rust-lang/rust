//! The AOT driver uses [`cranelift_object`] to write object files suitable for linking into a
//! standalone executable.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::{InlineAsmOperand, ItemId};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_session::config::{OutputFilenames, OutputType};
use rustc_target::asm::InlineAsmArch;

use crate::prelude::*;

pub(crate) fn codegen_global_asm_item(tcx: TyCtxt<'_>, global_asm: &mut String, item_id: ItemId) {
    let item = tcx.hir_item(item_id);
    if let rustc_hir::ItemKind::GlobalAsm { asm, .. } = item.kind {
        let is_x86 =
            matches!(tcx.sess.asm_arch.unwrap(), InlineAsmArch::X86 | InlineAsmArch::X86_64);

        if is_x86 {
            if !asm.options.contains(InlineAsmOptions::ATT_SYNTAX) {
                global_asm.push_str("\n.intel_syntax noprefix\n");
            } else {
                global_asm.push_str("\n.att_syntax\n");
            }
        }
        for piece in asm.template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => global_asm.push_str(s),
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier: _, span: op_sp } => {
                    match asm.operands[operand_idx].0 {
                        InlineAsmOperand::Const { ref anon_const } => {
                            match tcx.const_eval_poly(anon_const.def_id.to_def_id()) {
                                Ok(const_value) => {
                                    let ty = tcx
                                        .typeck_body(anon_const.body)
                                        .node_type(anon_const.hir_id);
                                    let string = rustc_codegen_ssa::common::asm_const_to_str(
                                        tcx,
                                        op_sp,
                                        const_value,
                                        FullyMonomorphizedLayoutCx(tcx).layout_of(ty),
                                    );
                                    global_asm.push_str(&string);
                                }
                                Err(ErrorHandled::Reported { .. }) => {
                                    // An error has already been reported and compilation is
                                    // guaranteed to fail if execution hits this path.
                                }
                                Err(ErrorHandled::TooGeneric(_)) => {
                                    span_bug!(op_sp, "asm const cannot be resolved; too generic");
                                }
                            }
                        }
                        InlineAsmOperand::SymFn { expr } => {
                            if cfg!(not(feature = "inline_asm_sym")) {
                                tcx.dcx().span_err(
                                    item.span,
                                    "asm! and global_asm! sym operands are not yet supported",
                                );
                            }

                            let ty = tcx.typeck(item_id.owner_id).expr_ty(expr);
                            let instance = match ty.kind() {
                                &ty::FnDef(def_id, args) => Instance::new(def_id, args),
                                _ => span_bug!(op_sp, "asm sym is not a function"),
                            };
                            let symbol = tcx.symbol_name(instance);
                            // FIXME handle the case where the function was made private to the
                            // current codegen unit
                            global_asm.push_str(symbol.name);
                        }
                        InlineAsmOperand::SymStatic { path: _, def_id } => {
                            if cfg!(not(feature = "inline_asm_sym")) {
                                tcx.dcx().span_err(
                                    item.span,
                                    "asm! and global_asm! sym operands are not yet supported",
                                );
                            }

                            let instance = Instance::mono(tcx, def_id);
                            let symbol = tcx.symbol_name(instance);
                            global_asm.push_str(symbol.name);
                        }
                        InlineAsmOperand::In { .. }
                        | InlineAsmOperand::Out { .. }
                        | InlineAsmOperand::InOut { .. }
                        | InlineAsmOperand::SplitInOut { .. }
                        | InlineAsmOperand::Label { .. } => {
                            span_bug!(op_sp, "invalid operand type for global_asm!")
                        }
                    }
                }
            }
        }

        global_asm.push('\n');
        if is_x86 {
            global_asm.push_str(".att_syntax\n\n");
        }
    } else {
        bug!("Expected GlobalAsm found {:?}", item);
    }
}

#[derive(Debug)]
pub(crate) struct GlobalAsmConfig {
    assembler: PathBuf,
    target: String,
    pub(crate) output_filenames: Arc<OutputFilenames>,
}

impl GlobalAsmConfig {
    pub(crate) fn new(tcx: TyCtxt<'_>) -> Self {
        GlobalAsmConfig {
            assembler: crate::toolchain::get_toolchain_binary(tcx.sess, "as"),
            target: match &tcx.sess.opts.target_triple {
                rustc_target::spec::TargetTuple::TargetTuple(triple) => triple.clone(),
                rustc_target::spec::TargetTuple::TargetJson { path_for_rustdoc, .. } => {
                    path_for_rustdoc.to_str().unwrap().to_owned()
                }
            },
            output_filenames: tcx.output_filenames(()).clone(),
        }
    }
}

pub(crate) fn compile_global_asm(
    config: &GlobalAsmConfig,
    cgu_name: &str,
    global_asm: &str,
    invocation_temp: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    if global_asm.is_empty() {
        return Ok(None);
    }

    // Remove all LLVM style comments
    let mut global_asm = global_asm
        .lines()
        .map(|line| if let Some(index) = line.find("//") { &line[0..index] } else { line })
        .collect::<Vec<_>>()
        .join("\n");
    global_asm.push('\n');

    let global_asm_object_file = add_file_stem_postfix(
        config.output_filenames.temp_path_for_cgu(OutputType::Object, cgu_name, invocation_temp),
        ".asm",
    );

    // Assemble `global_asm`
    if option_env!("CG_CLIF_FORCE_GNU_AS").is_some() {
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
    } else {
        let mut child = Command::new(std::env::current_exe().unwrap())
            // Avoid a warning about the jobserver fd not being passed
            .env_remove("CARGO_MAKEFLAGS")
            .arg("--target")
            .arg(&config.target)
            .arg("--crate-type")
            .arg("staticlib")
            .arg("--emit")
            .arg("obj")
            .arg("-o")
            .arg(&global_asm_object_file)
            .arg("-")
            .arg("-Abad_asm_style")
            .arg("-Zcodegen-backend=llvm")
            .stdin(Stdio::piped())
            .spawn()
            .expect("Failed to spawn `as`.");
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(
                br####"
                #![feature(decl_macro, no_core, rustc_attrs)]
                #![allow(internal_features)]
                #![no_core]
                #[rustc_builtin_macro]
                #[rustc_macro_transparency = "semitransparent"]
                macro global_asm() { /* compiler built-in */ }
                global_asm!(r###"
                "####,
            )
            .unwrap();
        stdin.write_all(global_asm.as_bytes()).unwrap();
        stdin
            .write_all(
                br####"
                "###);
                "####,
            )
            .unwrap();
        std::mem::drop(stdin);
        let status = child.wait().expect("Failed to wait for `as`.");
        if !status.success() {
            return Err(format!("Failed to assemble `{}`", global_asm));
        }
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
