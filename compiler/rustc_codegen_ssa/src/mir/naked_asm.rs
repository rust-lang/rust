use rustc_abi::{BackendRepr, Float, Integer, Primitive, RegKind};
use rustc_attr_parsing::InstructionSetAttr;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::mono::{Linkage, MonoItem, MonoItemData, Visibility};
use rustc_middle::mir::{Body, InlineAsmOperand};
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use rustc_middle::{bug, span_bug, ty};
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::WasmCAbi;

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
    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
    let (begin, end) = prefix_and_suffix(cx.tcx(), instance, &name, item_data, fn_abi);

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
    Wasm,
}

impl AsmBinaryFormat {
    fn from_target(target: &rustc_target::spec::Target) -> Self {
        if target.is_like_windows {
            Self::Coff
        } else if target.is_like_osx {
            Self::Macho
        } else if target.is_like_wasm {
            Self::Wasm
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
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> (String, String) {
    use std::fmt::Write;

    let asm_binary_format = AsmBinaryFormat::from_target(&tcx.sess.target);

    let is_arm = tcx.sess.target.arch == "arm";
    let is_thumb = tcx.sess.unstable_target_features.contains(&sym::thumb_mode);

    let attrs = tcx.codegen_fn_attrs(instance.def_id());
    let link_section = attrs.link_section.map(|symbol| symbol.as_str().to_string());

    // function alignment can be set globally with the `-Zmin-function-alignment=<n>` flag;
    // the alignment from a `#[repr(align(<n>))]` is used if it specifies a higher alignment.
    // if no alignment is specified, an alignment of 4 bytes is used.
    let min_function_alignment = tcx.sess.opts.unstable_opts.min_function_alignment;
    let align = Ord::max(min_function_alignment, attrs.alignment).map(|a| a.bytes()).unwrap_or(4);

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
                    AsmBinaryFormat::Elf | AsmBinaryFormat::Coff | AsmBinaryFormat::Wasm => {
                        writeln!(w, ".weak {asm_name}")?;
                    }
                    AsmBinaryFormat::Macho => {
                        writeln!(w, ".globl {asm_name}")?;
                        writeln!(w, ".weak_definition {asm_name}")?;
                    }
                }
            }
            Linkage::Internal | Linkage::Private => {
                // write nothing
            }
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
        }

        Ok(())
    };

    let mut begin = String::new();
    let mut end = String::new();
    match asm_binary_format {
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
            write_linkage(&mut begin).unwrap();
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
            write_linkage(&mut begin).unwrap();
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
            write_linkage(&mut begin).unwrap();
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
        AsmBinaryFormat::Wasm => {
            let section = link_section.unwrap_or(format!(".text.{asm_name}"));

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
            writeln!(
                begin,
                ".functype {asm_name} {}",
                wasm_functype(tcx, fn_abi, instance.def_id())
            )
            .unwrap();

            writeln!(end).unwrap();
            // .size is ignored for function symbols, so we can skip it
            writeln!(end, "end_function").unwrap();
        }
    }

    (begin, end)
}

/// The webassembly type signature for the given function.
///
/// Used by the `.functype` directive on wasm targets.
fn wasm_functype<'tcx>(tcx: TyCtxt<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>, def_id: DefId) -> String {
    let mut signature = String::with_capacity(64);

    let ptr_type = match tcx.data_layout.pointer_size.bits() {
        32 => "i32",
        64 => "i64",
        other => bug!("wasm pointer size cannot be {other} bits"),
    };

    // FIXME: remove this once the wasm32-unknown-unknown ABI is fixed
    // please also add `wasm32-unknown-unknown` back in `tests/assembly/wasm32-naked-fn.rs`
    // basically the commit introducing this comment should be reverted
    if let PassMode::Pair { .. } = fn_abi.ret.mode {
        let _ = WasmCAbi::Legacy;
        span_bug!(
            tcx.def_span(def_id),
            "cannot return a pair (the wasm32-unknown-unknown ABI is broken, see https://github.com/rust-lang/rust/issues/115666"
        );
    }

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
        wasm_type(tcx, &mut signature, arg_abi, ptr_type, def_id);
        if it.peek().is_some() {
            signature.push_str(", ");
        }
    }

    signature.push_str(") -> (");

    if !hidden_return {
        wasm_type(tcx, &mut signature, &fn_abi.ret, ptr_type, def_id);
    }

    signature.push(')');

    signature
}

fn wasm_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    signature: &mut String,
    arg_abi: &ArgAbi<'_, Ty<'tcx>>,
    ptr_type: &'static str,
    def_id: DefId,
) {
    match arg_abi.mode {
        PassMode::Ignore => { /* do nothing */ }
        PassMode::Direct(_) => {
            let direct_type = match arg_abi.layout.backend_repr {
                BackendRepr::Scalar(scalar) => wasm_primitive(scalar.primitive(), ptr_type),
                BackendRepr::Vector { .. } => "v128",
                BackendRepr::Memory { .. } => {
                    // FIXME: remove this branch once the wasm32-unknown-unknown ABI is fixed
                    let _ = WasmCAbi::Legacy;
                    span_bug!(
                        tcx.def_span(def_id),
                        "cannot use memory args (the wasm32-unknown-unknown ABI is broken, see https://github.com/rust-lang/rust/issues/115666"
                    );
                }
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
