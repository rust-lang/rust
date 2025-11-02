use rustc_abi::{BackendRepr, Float, Integer, Primitive, RegKind};
use rustc_hir::attrs::{InstructionSetAttr, Linkage};
use rustc_middle::mir::mono::{MonoItemData, Visibility};
use rustc_middle::mir::{InlineAsmOperand, START_BLOCK};
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_middle::ty::{Instance, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, ty};
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::BinaryFormat;

use crate::common;
use crate::mir::AsmCodegenMethods;
use crate::traits::GlobalAsmOperandRef;

pub fn codegen_naked_asm<
    'a,
    'tcx,
    Cx: LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>
        + FnAbiOf<'tcx, FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>>
        + AsmCodegenMethods<'tcx>,
>(
    cx: &'a mut Cx,
    instance: Instance<'tcx>,
    item_data: MonoItemData,
) {
    assert!(!instance.args.has_infer());
    let mir = cx.tcx().instance_mir(instance.def);

    let rustc_middle::mir::TerminatorKind::InlineAsm {
        asm_macro: _,
        template,
        ref operands,
        options,
        line_spans,
        targets: _,
        unwind: _,
    } = mir.basic_blocks[START_BLOCK].terminator().kind
    else {
        bug!("#[naked] functions should always terminate with an asm! block")
    };

    let operands: Vec<_> =
        operands.iter().map(|op| inline_to_global_operand::<Cx>(cx, instance, op)).collect();

    let name = cx.mangled_name(instance);
    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
    let (begin, end) = prefix_and_suffix(cx.tcx(), instance, &name, item_data, fn_abi);

    let mut template_vec = Vec::new();
    template_vec.push(rustc_ast::ast::InlineAsmTemplatePiece::String(begin.into()));
    template_vec.extend(template.iter().cloned());
    template_vec.push(rustc_ast::ast::InlineAsmTemplatePiece::String(end.into()));

    cx.codegen_global_asm(&template_vec, &operands, options, line_spans);
}

fn inline_to_global_operand<'a, 'tcx, Cx: LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>>(
    cx: &'a Cx,
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
                &ty::FnDef(def_id, args) => {
                    Instance::expect_resolve(cx.tcx(), cx.typing_env(), def_id, args, value.span)
                }
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

fn prefix_and_suffix<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    asm_name: &str,
    item_data: MonoItemData,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> (String, String) {
    use std::fmt::Write;

    let asm_binary_format = &tcx.sess.target.binary_format;

    let is_arm = tcx.sess.target.arch == "arm";
    let is_thumb = tcx.sess.unstable_target_features.contains(&sym::thumb_mode);

    let attrs = tcx.codegen_instance_attrs(instance.def);
    let link_section = attrs.link_section.map(|symbol| symbol.as_str().to_string());

    // If no alignment is specified, an alignment of 4 bytes is used.
    let align_bytes = attrs.alignment.map(|a| a.bytes()).unwrap_or(4);

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
    let write_linkage = |w: &mut String| -> std::fmt::Result {
        match item_data.linkage {
            Linkage::External => {
                writeln!(w, ".globl {asm_name}")?;
            }
            Linkage::LinkOnceAny | Linkage::LinkOnceODR | Linkage::WeakAny | Linkage::WeakODR => {
                match asm_binary_format {
                    BinaryFormat::Elf | BinaryFormat::Coff | BinaryFormat::Wasm => {
                        writeln!(w, ".weak {asm_name}")?;
                    }
                    BinaryFormat::Xcoff => {
                        // FIXME: there is currently no way of defining a weak symbol in inline assembly
                        // for AIX. See https://github.com/llvm/llvm-project/issues/130269
                        emit_fatal(
                            "cannot create weak symbols from inline assembly for this target",
                        )
                    }
                    BinaryFormat::MachO => {
                        writeln!(w, ".globl {asm_name}")?;
                        writeln!(w, ".weak_definition {asm_name}")?;
                    }
                }
            }
            Linkage::Internal => {
                // write nothing
            }
            Linkage::Common => emit_fatal("Functions may not have common linkage"),
            Linkage::AvailableExternally => {
                // this would make the function equal an extern definition
                emit_fatal("Functions may not have available_externally linkage")
            }
            Linkage::ExternalWeak => {
                // FIXME: actually this causes a SIGILL in LLVM
                emit_fatal("Functions may not have external weak linkage")
            }
        }

        Ok(())
    };

    let mut begin = String::new();
    let mut end = String::new();
    match asm_binary_format {
        BinaryFormat::Elf => {
            let section = link_section.unwrap_or_else(|| format!(".text.{asm_name}"));

            let progbits = match is_arm {
                true => "%progbits",
                false => "@progbits",
            };

            let function = match is_arm {
                true => "%function",
                false => "@function",
            };

            writeln!(begin, ".pushsection {section},\"ax\", {progbits}").unwrap();
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            match item_data.visibility {
                Visibility::Default => {}
                Visibility::Protected => writeln!(begin, ".protected {asm_name}").unwrap(),
                Visibility::Hidden => writeln!(begin, ".hidden {asm_name}").unwrap(),
            }
            writeln!(begin, ".type {asm_name}, {function}").unwrap();
            if !arch_prefix.is_empty() {
                writeln!(begin, "{}", arch_prefix).unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            // emit a label starting with `func_end` for `cargo asm` and other tooling that might
            // pattern match on assembly generated by LLVM.
            writeln!(end, ".Lfunc_end_{asm_name}:").unwrap();
            writeln!(end, ".size {asm_name}, . - {asm_name}").unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::MachO => {
            let section = link_section.unwrap_or_else(|| "__TEXT,__text".to_string());
            writeln!(begin, ".pushsection {},regular,pure_instructions", section).unwrap();
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            match item_data.visibility {
                Visibility::Default | Visibility::Protected => {}
                Visibility::Hidden => writeln!(begin, ".private_extern {asm_name}").unwrap(),
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            writeln!(end, ".Lfunc_end_{asm_name}:").unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::Coff => {
            let section = link_section.unwrap_or_else(|| format!(".text.{asm_name}"));
            writeln!(begin, ".pushsection {},\"xr\"", section).unwrap();
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            writeln!(begin, ".def {asm_name}").unwrap();
            writeln!(begin, ".scl 2").unwrap();
            writeln!(begin, ".type 32").unwrap();
            writeln!(begin, ".endef").unwrap();
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            writeln!(end, ".Lfunc_end_{asm_name}:").unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::Wasm => {
            let section = link_section.unwrap_or_else(|| format!(".text.{asm_name}"));

            writeln!(begin, ".section {section},\"\",@").unwrap();
            // wasm functions cannot be aligned, so skip
            write_linkage(&mut begin).unwrap();
            if let Visibility::Hidden = item_data.visibility {
                writeln!(begin, ".hidden {asm_name}").unwrap();
            }
            writeln!(begin, ".type {asm_name}, @function").unwrap();
            if !arch_prefix.is_empty() {
                writeln!(begin, "{}", arch_prefix).unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();
            writeln!(begin, ".functype {asm_name} {}", wasm_functype(tcx, fn_abi)).unwrap();

            writeln!(end).unwrap();
            // .size is ignored for function symbols, so we can skip it
            writeln!(end, "end_function").unwrap();
            writeln!(end, ".Lfunc_end_{asm_name}:").unwrap();
        }
        BinaryFormat::Xcoff => {
            // the LLVM XCOFFAsmParser is extremely incomplete and does not implement many of the
            // documented directives.
            //
            // - https://github.com/llvm/llvm-project/blob/1b25c0c4da968fe78921ce77736e5baef4db75e3/llvm/lib/MC/MCParser/XCOFFAsmParser.cpp
            // - https://www.ibm.com/docs/en/ssw_aix_71/assembler/assembler_pdf.pdf
            //
            // Consequently, we try our best here but cannot do as good a job as for other binary
            // formats.

            // FIXME: start a section. `.csect` is not currently implemented in LLVM

            // fun fact: according to the assembler documentation, .align takes an exponent,
            // but LLVM only accepts powers of 2 (but does emit the exponent)
            // so when we hand `.align 32` to LLVM, the assembly output will contain `.align 5`
            writeln!(begin, ".align {}", align_bytes).unwrap();

            write_linkage(&mut begin).unwrap();
            if let Visibility::Hidden = item_data.visibility {
                // FIXME apparently `.globl {asm_name}, hidden` is valid
                // but due to limitations with `.weak` (see above) we can't really use that in general yet
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            // FIXME: end the section?
        }
    }

    (begin, end)
}

/// The webassembly type signature for the given function.
///
/// Used by the `.functype` directive on wasm targets.
fn wasm_functype<'tcx>(tcx: TyCtxt<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> String {
    let mut signature = String::with_capacity(64);

    let ptr_type = match tcx.data_layout.pointer_size().bits() {
        32 => "i32",
        64 => "i64",
        other => bug!("wasm pointer size cannot be {other} bits"),
    };

    let hidden_return = matches!(fn_abi.ret.mode, PassMode::Indirect { .. });

    signature.push('(');

    if hidden_return {
        signature.push_str(ptr_type);
        if !fn_abi.args.is_empty() {
            signature.push_str(", ");
        }
    }

    let mut it = fn_abi.args.iter().peekable();
    while let Some(arg_abi) = it.next() {
        wasm_type(&mut signature, arg_abi, ptr_type);
        if it.peek().is_some() {
            signature.push_str(", ");
        }
    }

    signature.push_str(") -> (");

    if !hidden_return {
        wasm_type(&mut signature, &fn_abi.ret, ptr_type);
    }

    signature.push(')');

    signature
}

fn wasm_type<'tcx>(signature: &mut String, arg_abi: &ArgAbi<'_, Ty<'tcx>>, ptr_type: &'static str) {
    match arg_abi.mode {
        PassMode::Ignore => { /* do nothing */ }
        PassMode::Direct(_) => {
            let direct_type = match arg_abi.layout.backend_repr {
                BackendRepr::Scalar(scalar) => wasm_primitive(scalar.primitive(), ptr_type),
                BackendRepr::SimdVector { .. } => "v128",
                other => unreachable!("unexpected BackendRepr: {:?}", other),
            };

            signature.push_str(direct_type);
        }
        PassMode::Pair(_, _) => match arg_abi.layout.backend_repr {
            BackendRepr::ScalarPair(a, b) => {
                signature.push_str(wasm_primitive(a.primitive(), ptr_type));
                signature.push_str(", ");
                signature.push_str(wasm_primitive(b.primitive(), ptr_type));
            }
            other => unreachable!("{other:?}"),
        },
        PassMode::Cast { pad_i32, ref cast } => {
            // For wasm, Cast is used for single-field primitive wrappers like `struct Wrapper(i64);`
            assert!(!pad_i32, "not currently used by wasm calling convention");
            assert!(cast.prefix[0].is_none(), "no prefix");
            assert_eq!(cast.rest.total, arg_abi.layout.size, "single item");

            let wrapped_wasm_type = match cast.rest.unit.kind {
                RegKind::Integer => match cast.rest.unit.size.bytes() {
                    ..=4 => "i32",
                    ..=8 => "i64",
                    _ => ptr_type,
                },
                RegKind::Float => match cast.rest.unit.size.bytes() {
                    ..=4 => "f32",
                    ..=8 => "f64",
                    _ => ptr_type,
                },
                RegKind::Vector => "v128",
            };

            signature.push_str(wrapped_wasm_type);
        }
        PassMode::Indirect { .. } => signature.push_str(ptr_type),
    }
}

fn wasm_primitive(primitive: Primitive, ptr_type: &'static str) -> &'static str {
    match primitive {
        Primitive::Int(integer, _) => match integer {
            Integer::I8 | Integer::I16 | Integer::I32 => "i32",
            Integer::I64 => "i64",
            Integer::I128 => "i64, i64",
        },
        Primitive::Float(float) => match float {
            Float::F16 | Float::F32 => "f32",
            Float::F64 => "f64",
            Float::F128 => "i64, i64",
        },
        Primitive::Pointer(_) => ptr_type,
    }
}
