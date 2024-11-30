use rustc_attr::InstructionSetAttr;
use rustc_middle::mir::mono::{Linkage, MonoItem, MonoItemData, Visibility};
use rustc_middle::mir::{Body, InlineAsmOperand};
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_span::sym;

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
    let name = cx.mangled_name(instance);
    let (begin, end) = prefix_and_suffix(cx.tcx(), instance, &name, item_data);

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
        } else if target.is_like_osx {
            Self::Macho
        } else {
            Self::Elf
        }
    }
}

fn prefix_and_suffix<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    asm_name: &str,
    item_data: &MonoItemData,
) -> (String, String) {
    use std::fmt::Write;

    let is_arm = tcx.sess.target.arch == "arm";
    let is_thumb = tcx.sess.unstable_target_features.contains(&sym::thumb_mode);

    let attrs = tcx.codegen_fn_attrs(instance.def_id());
    let link_section = attrs.link_section.map(|symbol| symbol.as_str().to_string());
    let align = attrs.alignment.map(|a| a.bytes()).unwrap_or(4);

    // See https://sourceware.org/binutils/docs/as/ARM-Directives.html for info on these directives.
    // In particular, `.arm` can also be written `.code 32` and `.thumb` as `.code 16`.
    let (arch_prefix, arch_suffix) = if is_arm {
        (
            match attrs.instruction_set {
                None => match is_thumb {
                    true => ".thumb\n.thumb_func",
                    false => ".arm",
                },
                Some(InstructionSetAttr::ArmT32) => ".thumb\n.thumb_func",
                Some(InstructionSetAttr::ArmA32) => ".arm",
            },
            match is_thumb {
                true => ".thumb",
                false => ".arm",
            },
        )
    } else {
        ("", "")
    };

    let emit_fatal = |msg| tcx.dcx().span_fatal(tcx.def_span(instance.def_id()), msg);

    // see https://godbolt.org/z/cPK4sxKor.
    // None means the default, which corresponds to internal linkage
    let linkage = match item_data.linkage {
        Linkage::External => Some(".globl"),
        Linkage::LinkOnceAny => Some(".weak"),
        Linkage::LinkOnceODR => Some(".weak"),
        Linkage::WeakAny => Some(".weak"),
        Linkage::WeakODR => Some(".weak"),
        Linkage::Internal => None,
        Linkage::Private => None,
        Linkage::Appending => emit_fatal("Only global variables can have appending linkage!"),
        Linkage::Common => emit_fatal("Functions may not have common linkage"),
        Linkage::AvailableExternally => {
            // this would make the function equal an extern definition
            emit_fatal("Functions may not have available_externally linkage")
        }
        Linkage::ExternalWeak => {
            // FIXME: actually this causes a SIGILL in LLVM
            emit_fatal("Functions may not have external weak linkage")
        }
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
            writeln!(begin, ".balign {align}").unwrap();
            if let Some(linkage) = linkage {
                writeln!(begin, "{linkage} {asm_name}").unwrap();
            }
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
            writeln!(begin, ".balign {align}").unwrap();
            if let Some(linkage) = linkage {
                writeln!(begin, "{linkage} {asm_name}").unwrap();
            }
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
            writeln!(begin, ".balign {align}").unwrap();
            if let Some(linkage) = linkage {
                writeln!(begin, "{linkage} {asm_name}").unwrap();
            }
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
