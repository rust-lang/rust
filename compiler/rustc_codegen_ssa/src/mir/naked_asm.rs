use object::{Architecture, SubArchitecture};
use rustc_abi::{BackendRepr, Float, Integer, Primitive, RegKind};
use rustc_attr_parsing::InstructionSetAttr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrs, TargetFeature};
use rustc_middle::mir::mono::{Linkage, MonoItem, MonoItemData, Visibility};
use rustc_middle::mir::{Body, InlineAsmOperand};
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use rustc_middle::{bug, span_bug, ty};
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::{BinaryFormat, WasmCAbi};

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

// FIXME share code with `create_object_file`
fn parse_architecture(
    sess: &rustc_session::Session,
) -> Option<(Architecture, Option<SubArchitecture>)> {
    let (architecture, subarchitecture) = match &sess.target.arch[..] {
        "arm" => (Architecture::Arm, None),
        "aarch64" => (
            if sess.target.pointer_width == 32 {
                Architecture::Aarch64_Ilp32
            } else {
                Architecture::Aarch64
            },
            None,
        ),
        "x86" => (Architecture::I386, None),
        "s390x" => (Architecture::S390x, None),
        "mips" | "mips32r6" => (Architecture::Mips, None),
        "mips64" | "mips64r6" => (Architecture::Mips64, None),
        "x86_64" => (
            if sess.target.pointer_width == 32 {
                Architecture::X86_64_X32
            } else {
                Architecture::X86_64
            },
            None,
        ),
        "powerpc" => (Architecture::PowerPc, None),
        "powerpc64" => (Architecture::PowerPc64, None),
        "riscv32" => (Architecture::Riscv32, None),
        "riscv64" => (Architecture::Riscv64, None),
        "sparc" => {
            if sess.unstable_target_features.contains(&sym::v8plus) {
                // Target uses V8+, aka EM_SPARC32PLUS, aka 64-bit V9 but in 32-bit mode
                (Architecture::Sparc32Plus, None)
            } else {
                // Target uses V7 or V8, aka EM_SPARC
                (Architecture::Sparc, None)
            }
        }
        "sparc64" => (Architecture::Sparc64, None),
        "avr" => (Architecture::Avr, None),
        "msp430" => (Architecture::Msp430, None),
        "hexagon" => (Architecture::Hexagon, None),
        "bpf" => (Architecture::Bpf, None),
        "loongarch64" => (Architecture::LoongArch64, None),
        "csky" => (Architecture::Csky, None),
        "arm64ec" => (Architecture::Aarch64, Some(SubArchitecture::Arm64EC)),

        // These architecutres are added here, and not present in `create_object_file`
        "wasm32" => (Architecture::Wasm32, None),
        "wasm64" => (Architecture::Wasm64, None),
        "m68k" => (Architecture::M68k, None),

        // Unsupported architecture.
        _ => return None,
    };

    Some((architecture, subarchitecture))
}

/// Enable the function's target features in the body of the function, then disable them again
fn enable_disable_target_features<'tcx>(
    tcx: TyCtxt<'tcx>,
    attrs: &CodegenFnAttrs,
) -> Option<(String, String)> {
    use std::fmt::Write;

    let mut begin = String::new();
    let mut end = String::new();

    let (architecture, _subarchitecture) = parse_architecture(tcx.sess)?;
    let features = attrs.target_features.iter().filter(|attr| !attr.implied);

    match architecture {
        Architecture::X86_64 | Architecture::X86_64_X32 | Architecture::I386 => {
            // no action is needed, all instructions are accepted regardless of target feature
        }

        Architecture::Aarch64 | Architecture::Aarch64_Ilp32 => {
            // https://developer.arm.com/documentation/100067/0611/armclang-Integrated-Assembler/AArch32-Target-selection-directives?lang=en

            for feature in features {
                // only enable/disable a feature if it is not already globally enabled.
                // so that we don't infuence subsequent asm blocks
                if !tcx.sess.unstable_target_features.contains(&feature.name) {
                    writeln!(begin, ".arch_extension {}", feature.name).unwrap();

                    writeln!(end, ".arch_extension no{}", feature.name).unwrap();
                }
            }
        }
        Architecture::Arm => {
            // https://developer.arm.com/documentation/100067/0611/armclang-Integrated-Assembler/AArch32-Target-selection-directives?lang=en

            // FIXME: implement the target feature handling. Contrary to most other targets, there
            // is no convenient push/pop mechanism. That means we have to manually restore the state
            // of the target features to the state on entry. This is complicated, given how arm
            // features interact. There is some incomplete code here https://gist.github.com/folkertdev/fe99874c466e598d0fb2dadf13b91b6f

            /* fallthrough */
        }

        Architecture::Riscv32 | Architecture::Riscv64 => {
            // https://github.com/riscv-non-isa/riscv-asm-manual/blob/ad0de8c004e29c9a7ac33cfd054f4d4f9392f2fb/src/asm-manual.adoc#arch

            writeln!(begin, ".option push").unwrap();
            for feature in features {
                writeln!(begin, ".option arch, +{}", feature.name).unwrap();
            }

            writeln!(end, ".option pop").unwrap();
        }
        Architecture::Mips | Architecture::Mips64 | Architecture::Mips64_N32 => {
            // https://sourceware.org/binutils/docs/as/MIPS-ISA.html
            // https://sourceware.org/binutils/docs/as/MIPS-ASE-Instruction-Generation-Overrides.html

            writeln!(begin, ".set push").unwrap();
            for feature in features {
                writeln!(begin, ".set {}", feature.name).unwrap();
            }

            writeln!(end, ".set pop").unwrap();
        }

        Architecture::S390x => {
            // https://sourceware.org/binutils/docs/as/s390-Directives.html

            // based on src/llvm-project/llvm/lib/Target/SystemZ/SystemZFeatures.td
            let isa_revision_for_feature_name = |feature_name| match feature_name {
                "backchain" => None, // does not define any instructions
                "deflate-conversion" => Some(13),
                "enhanced-sort" => Some(13),
                "guarded-storage" => Some(12),
                "high-word" => None, // technically 9, but LLVM supports only >= 10
                "nnp-assist" => Some(14),
                "transactional-execution" => Some(10),
                "vector" => Some(11),
                "vector-enhancements-1" => Some(12),
                "vector-enhancements-2" => Some(13),
                "vector-packed-decimal" => Some(12),
                "vector-packed-decimal-enhancement" => Some(13),
                "vector-packed-decimal-enhancement-2" => Some(14),
                _ => None,
            };

            let target_feature_isa = features
                .filter_map(|feature| isa_revision_for_feature_name(feature.name.as_str()))
                .max();

            if let Some(minimum_isa) = target_feature_isa {
                writeln!(begin, ".machine arch{minimum_isa}").unwrap();

                // NOTE: LLVM does not currently support `.machine push` and `.machine pop`
                // this is tracked in https://github.com/llvm/llvm-project/issues/129053.
                //
                // So instead we have to try revert to the previous state manually.
                //
                // However, this may still be observable if the user explicitly set the machine to
                // a higher value using global assembly.
                let global_isa = tcx
                    .sess
                    .unstable_target_features
                    .iter()
                    .filter_map(|feature| isa_revision_for_feature_name(feature.as_str()))
                    .max()
                    .unwrap_or(10);

                writeln!(end, ".machine arch{global_isa}").unwrap();
            }
        }
        Architecture::PowerPc | Architecture::PowerPc64 => {
            // https://www.ibm.com/docs/en/ssw_aix_71/assembler/assembler_pdf.pdf

            // based on src/llvm-project/llvm/lib/Target/PowerPC/PPC.td
            let isa_revision_for_feature = |feature: &TargetFeature| match feature.name.as_str() {
                "altivec" => Some(7),
                "partword-atomics" => Some(8),
                "power10-vector" => Some(10),
                "power8-altivec" => Some(8),
                "power8-crypto" => Some(8),
                "power8-vector" => Some(9),
                "power9-altivec" => Some(9),
                "power9-vector" => Some(9),
                "quadword-atomics" => Some(8),
                "vsx" => Some(7),
                _ => None,
            };

            if let Some(minimum_isa) = features.filter_map(isa_revision_for_feature).max() {
                writeln!(begin, ".machine push").unwrap();

                // LLVM currently ignores the .machine directive, and allows all instructions regardless
                // of the machine. This may be fixed in the future.
                //
                // https://github.com/llvm/llvm-project/blob/74306afe87b85cb9b5734044eb6c74b8290098b3/llvm/lib/Target/PowerPC/AsmParser/PPCAsmParser.cpp#L1799
                writeln!(begin, ".machine pwr{minimum_isa}").unwrap();

                writeln!(end, ".machine pop").unwrap();
            }
        }

        Architecture::M68k => {
            // https://sourceware.org/binutils/docs/as/M68K_002dDirectives.html#index-directives_002c-M680x0

            // M68k suports the .cpu and .arch directives, but they both can only be applied once
            //
            // > If it is given multiple times, or in conjunction with the -march option,
            // > all uses must be for the same architecture and extension set.
            //
            // That is not flexible enough for us, because different functions might want different
            // features.
            //
            // So far, we've not found any cases where ignoring the target features causes issues,
            // so that's what we do for now.

            /* fallthrough */
        }

        Architecture::Wasm32 | Architecture::Wasm64 => {
            // LLVM does not appear to accept any directive to enable target features
            //
            // https://github.com/llvm/llvm-project/blob/74306afe87b85cb9b5734044eb6c74b8290098b3/llvm/lib/Target/WebAssembly/AsmParser/WebAssemblyAsmParser.cpp#L909

            /* fallthrough */
        }

        Architecture::LoongArch64 => {
            // LLVM does not appear to accept any directive to enable target features
            //
            // https://github.com/llvm/llvm-project/blob/74306afe87b85cb9b5734044eb6c74b8290098b3/llvm/lib/Target/LoongArch/AsmParser/LoongArchAsmParser.cpp#L1918

            /* fallthrough */
        }

        // FIXME: support naked_asm! on more architectures
        Architecture::Avr => return None,
        Architecture::Bpf => return None,
        Architecture::Csky => return None,
        Architecture::E2K32 => return None,
        Architecture::E2K64 => return None,
        Architecture::Hexagon => return None,
        Architecture::Msp430 => return None,
        Architecture::Sbf => return None,
        Architecture::Sharc => return None,
        Architecture::Sparc => return None,
        Architecture::Sparc32Plus => return None,
        Architecture::Sparc64 => return None,
        Architecture::Xtensa => return None,

        // the Architecture enum is non-exhaustive
        Architecture::Unknown | _ => return None,
    }

    Some((begin, end))
}

fn prefix_and_suffix<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    asm_name: &str,
    item_data: &MonoItemData,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> (String, String) {
    use std::fmt::Write;

    let asm_binary_format = &tcx.sess.target.binary_format;

    let is_arm = tcx.sess.target.arch == "arm";
    let is_thumb = tcx.sess.unstable_target_features.contains(&sym::thumb_mode);

    let attrs = tcx.codegen_fn_attrs(instance.def_id());
    let link_section = attrs.link_section.map(|symbol| symbol.as_str().to_string());

    // function alignment can be set globally with the `-Zmin-function-alignment=<n>` flag;
    // the alignment from a `#[repr(align(<n>))]` is used if it specifies a higher alignment.
    // if no alignment is specified, an alignment of 4 bytes is used.
    let min_function_alignment = tcx.sess.opts.unstable_opts.min_function_alignment;
    let align_bytes =
        Ord::max(min_function_alignment, attrs.alignment).map(|a| a.bytes()).unwrap_or(4);

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

    let Some((target_feature_begin, target_feature_end)) =
        enable_disable_target_features(tcx, attrs)
    else {
        panic!("target features on naked functions are not supported for this architecture");
    };

    let mut begin = String::new();
    let mut end = String::new();
    match asm_binary_format {
        BinaryFormat::Elf => {
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
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            begin.push_str(&target_feature_begin);

            if let Visibility::Hidden = item_data.visibility {
                writeln!(begin, ".hidden {asm_name}").unwrap();
            }
            writeln!(begin, ".type {asm_name}, {function}").unwrap();
            if !arch_prefix.is_empty() {
                writeln!(begin, "{}", arch_prefix).unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            end.push_str(&target_feature_end);
            writeln!(end, ".size {asm_name}, . - {asm_name}").unwrap();
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::MachO => {
            let section = link_section.unwrap_or("__TEXT,__text".to_string());
            writeln!(begin, ".pushsection {},regular,pure_instructions", section).unwrap();
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            begin.push_str(&target_feature_begin);
            if let Visibility::Hidden = item_data.visibility {
                writeln!(begin, ".private_extern {asm_name}").unwrap();
            }
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            end.push_str(&target_feature_end);
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::Coff => {
            let section = link_section.unwrap_or(format!(".text.{asm_name}"));
            writeln!(begin, ".pushsection {},\"xr\"", section).unwrap();
            writeln!(begin, ".balign {align_bytes}").unwrap();
            write_linkage(&mut begin).unwrap();
            begin.push_str(&target_feature_begin);
            writeln!(begin, ".def {asm_name}").unwrap();
            writeln!(begin, ".scl 2").unwrap();
            writeln!(begin, ".type 32").unwrap();
            writeln!(begin, ".endef").unwrap();
            writeln!(begin, "{asm_name}:").unwrap();

            writeln!(end).unwrap();
            end.push_str(&target_feature_end);
            writeln!(end, ".popsection").unwrap();
            if !arch_suffix.is_empty() {
                writeln!(end, "{}", arch_suffix).unwrap();
            }
        }
        BinaryFormat::Wasm => {
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
        let _ = WasmCAbi::Legacy { with_lint: true };
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
                BackendRepr::SimdVector { .. } => "v128",
                BackendRepr::Memory { .. } => {
                    // FIXME: remove this branch once the wasm32-unknown-unknown ABI is fixed
                    let _ = WasmCAbi::Legacy { with_lint: true };
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
