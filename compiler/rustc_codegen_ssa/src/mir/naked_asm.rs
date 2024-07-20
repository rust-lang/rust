use rustc_attr::InstructionSetAttr;
use rustc_middle::mir::mono::{MonoItem, MonoItemData, Visibility};
use rustc_middle::mir::{Body, InlineAsmOperand};
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_target::asm::InlineAsmArch;

use crate::common;
use crate::traits::{AsmCodegenMethods, BuilderMethods, GlobalAsmOperandRef, MiscCodegenMethods};

pub(crate) fn codegen_naked_asm<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    mir: &Body<'tcx>,
    instance: Instance<'tcx>,
) {
    let rustc_middle::mir::TerminatorKind::InlineAsm {
        asm_macro: _,
        template,
        ref operands,
        options,
        line_spans,
        targets: _,
        unwind: _,
    } = mir.basic_blocks.iter().next().unwrap().terminator().kind
    else {
        bug!("#[naked] functions should always terminate with an asm! block")
    };

    let operands: Vec<_> =
        operands.iter().map(|op| inline_to_global_operand::<Bx>(cx, instance, op)).collect();

    let item_data = cx.codegen_unit().items().get(&MonoItem::Fn(instance)).unwrap();
    let (begin, end) = crate::mir::naked_asm::prefix_and_suffix(cx.tcx(), instance, item_data);

    let mut template_vec = Vec::new();
    template_vec.push(rustc_ast::ast::InlineAsmTemplatePiece::String(begin.into()));
    template_vec.extend(template.iter().cloned());
    template_vec.push(rustc_ast::ast::InlineAsmTemplatePiece::String(end.into()));

    cx.codegen_global_asm(&template_vec, &operands, options, line_spans);
}

fn inline_to_global_operand<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
    op: &InlineAsmOperand<'tcx>,
) -> GlobalAsmOperandRef<'tcx> {
    match op {
        InlineAsmOperand::Const { value } => {
            let const_value = instance
                .instantiate_mir_and_normalize_erasing_regions(
                    cx.tcx(),
                    cx.typing_env(),
                    ty::EarlyBinder::bind(value.const_),
                )
                .eval(cx.tcx(), cx.typing_env(), value.span)
                .expect("erroneous constant missed by mono item collection");

            let mono_type = instance.instantiate_mir_and_normalize_erasing_regions(
                cx.tcx(),
                cx.typing_env(),
                ty::EarlyBinder::bind(value.ty()),
            );

            let string = common::asm_const_to_str(
                cx.tcx(),
                value.span,
                const_value,
                cx.layout_of(mono_type),
            );

            GlobalAsmOperandRef::Const { string }
        }
        InlineAsmOperand::SymFn { value } => {
            let mono_type = instance.instantiate_mir_and_normalize_erasing_regions(
                cx.tcx(),
                cx.typing_env(),
                ty::EarlyBinder::bind(value.ty()),
            );

            let instance = match mono_type.kind() {
                &ty::FnDef(def_id, args) => Instance::new(def_id, args),
                _ => bug!("asm sym is not a function"),
            };

            GlobalAsmOperandRef::SymFn { instance }
        }
        InlineAsmOperand::SymStatic { def_id } => {
            GlobalAsmOperandRef::SymStatic { def_id: *def_id }
        }
        InlineAsmOperand::In { .. }
        | InlineAsmOperand::Out { .. }
        | InlineAsmOperand::InOut { .. }
        | InlineAsmOperand::Label { .. } => {
            bug!("invalid operand type for naked_asm!")
        }
    }
}

enum AsmBinaryFormat {
    Elf,
    Macho,
    Coff,
}

impl AsmBinaryFormat {
    fn from_target(target: &rustc_target::spec::Target) -> Self {
        if target.is_like_windows {
            Self::Coff
        } else if target.options.vendor == "apple" {
            Self::Macho
        } else {
            Self::Elf
        }
    }
}

fn prefix_and_suffix<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    item_data: &MonoItemData,
) -> (String, String) {
    use std::fmt::Write;

    let target = &tcx.sess.target;
    let target_arch = tcx.sess.asm_arch;

    let is_arm = target.arch == "arm";
    let is_thumb = is_arm && target.llvm_target.contains("thumb");

    let mangle = (target.is_like_windows && matches!(target_arch, Some(InlineAsmArch::X86)))
        || target.options.vendor == "apple";

    let asm_name = format!("{}{}", if mangle { "_" } else { "" }, tcx.symbol_name(instance).name);

    let attrs = tcx.codegen_fn_attrs(instance.def_id());
    let link_section = attrs.link_section.map(|symbol| symbol.as_str().to_string());

    let (arch_prefix, arch_suffix) = if is_arm {
        (
            match attrs.instruction_set {
                None => match is_thumb {
                    true => ".thumb\n.thumb_func",
                    false => ".arm",
                },
                Some(InstructionSetAttr::ArmA32) => ".arm",
                Some(InstructionSetAttr::ArmT32) => ".thumb\n.thumb_func",
            },
            match is_thumb {
                true => ".thumb",
                false => ".arm",
            },
        )
    } else {
        ("", "")
    };

    let mut begin = String::new();
    let mut end = String::new();
    match AsmBinaryFormat::from_target(&tcx.sess.target) {
        AsmBinaryFormat::Elf => {
            let section = link_section.unwrap_or(format!(".text.{asm_name}"));

            let progbits = match is_arm {
                true => "%progbits",
                false => "@progbits",
            };

            let function = match is_arm {
                true => "%function",
                false => "@function",
            };

            writeln!(begin, ".pushsection {section},\"ax\", {progbits}").unwrap();
            writeln!(begin, ".balign 4").unwrap();
            writeln!(begin, ".globl {asm_name}").unwrap();
            if let Visibility::Hidden = item_data.visibility {
                writeln!(begin, ".hidden {asm_name}").unwrap();
            }
            writeln!(begin, ".type {asm_name}, {function}").unwrap();
            if !arch_prefix.is_empty() {
                writeln!(begin, "{}", arch_prefix).unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            writeln!(end, ".size {asm_name}, . - {asm_name}").unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        AsmBinaryFormat::Macho => {
            let section = link_section.unwrap_or("__TEXT,__text".to_string());
            writeln!(begin, ".pushsection {},regular,pure_instructions", section).unwrap();
            writeln!(begin, ".balign 4").unwrap();
            writeln!(begin, ".globl {asm_name}").unwrap();
            if let Visibility::Hidden = item_data.visibility {
                writeln!(begin, ".private_extern {asm_name}").unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        AsmBinaryFormat::Coff => {
            let section = link_section.unwrap_or(format!(".text.{asm_name}"));
            writeln!(begin, ".pushsection {},\"xr\"", section).unwrap();
            writeln!(begin, ".balign 4").unwrap();
            writeln!(begin, ".globl {asm_name}").unwrap();
            writeln!(begin, ".def {asm_name}").unwrap();
            writeln!(begin, ".scl 2").unwrap();
            writeln!(begin, ".type 32").unwrap();
            writeln!(begin, ".endef {asm_name}").unwrap();
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
    }

    (begin, end)
}
