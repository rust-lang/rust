//! Codegen of a single function

use cranelift_codegen::CodegenError;
use cranelift_codegen::ir::UserFuncName;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::ModuleError;
use rustc_ast::InlineAsmOptions;
use rustc_codegen_ssa::base::is_call_from_compiler_builtins_to_upstream_monomorphization;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_index::IndexVec;
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv};
use rustc_middle::ty::print::with_no_trimmed_paths;

use crate::constant::ConstantCx;
use crate::debuginfo::{FunctionDebugContext, TypeDebugContext};
use crate::prelude::*;
use crate::pretty_clif::CommentWriter;
use crate::{codegen_f16_f128, enable_verifier};

pub(crate) struct CodegenedFunction {
    symbol_name: String,
    func_id: FuncId,
    func: Function,
    clif_comments: CommentWriter,
    func_debug_cx: Option<FunctionDebugContext>,
}

pub(crate) fn codegen_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut crate::CodegenCx,
    type_dbg: &mut TypeDebugContext<'tcx>,
    cached_func: Function,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) -> CodegenedFunction {
    debug_assert!(!instance.args.has_infer());

    let symbol_name = tcx.symbol_name(instance).name.to_string();
    let _timer = tcx.prof.generic_activity_with_arg("codegen fn", &*symbol_name);

    let mir = tcx.instance_mir(instance.def);
    let _mir_guard = crate::PrintOnPanic(|| {
        let mut buf = Vec::new();
        with_no_trimmed_paths!({
            let writer = pretty::MirWriter::new(tcx);
            writer.write_mir_fn(mir, &mut buf).unwrap();
        });
        String::from_utf8_lossy(&buf).into_owned()
    });

    // Declare function
    let sig = get_function_sig(tcx, module.target_config().default_call_conv, instance);
    let func_id = module.declare_function(&symbol_name, Linkage::Local, &sig).unwrap();

    // Make the FunctionBuilder
    let mut func_ctx = FunctionBuilderContext::new();
    let mut func = cached_func;
    func.clear();
    func.name = UserFuncName::user(0, func_id.as_u32());
    func.signature = sig;
    func.collect_debug_info();

    let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Predefine blocks
    let start_block = bcx.create_block();
    let block_map: IndexVec<BasicBlock, Block> =
        (0..mir.basic_blocks.len()).map(|_| bcx.create_block()).collect();

    let fn_abi = FullyMonomorphizedLayoutCx(tcx).fn_abi_of_instance(instance, ty::List::empty());

    // Make FunctionCx
    let target_config = module.target_config();
    let pointer_type = target_config.pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance, fn_abi);

    let func_debug_cx = if let Some(debug_context) = &mut cx.debug_context {
        Some(debug_context.define_function(tcx, type_dbg, instance, fn_abi, &symbol_name, mir.span))
    } else {
        None
    };

    let mut fx = FunctionCx {
        cx,
        module,
        tcx,
        target_config,
        pointer_type,
        constants_cx: ConstantCx::new(),
        func_debug_cx,

        instance,
        symbol_name,
        mir,
        fn_abi,

        bcx,
        block_map,
        local_map: IndexVec::with_capacity(mir.local_decls.len()),
        caller_location: None, // set by `codegen_fn_prelude`

        clif_comments,
        next_ssa_var: 0,
    };

    tcx.prof.generic_activity("codegen clif ir").run(|| codegen_fn_body(&mut fx, start_block));
    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();

    // Recover all necessary data from fx, before accessing func will prevent future access to it.
    let symbol_name = fx.symbol_name;
    let clif_comments = fx.clif_comments;
    let func_debug_cx = fx.func_debug_cx;

    fx.constants_cx.finalize(fx.tcx, &mut *fx.module);

    if cx.should_write_ir {
        crate::pretty_clif::write_clif_file(
            tcx.output_filenames(()),
            &symbol_name,
            "unopt",
            module.isa(),
            &func,
            &clif_comments,
        );
    }

    // Verify function
    verify_func(tcx, &clif_comments, &func);

    CodegenedFunction { symbol_name, func_id, func, clif_comments, func_debug_cx }
}

pub(crate) fn compile_fn(
    cx: &mut crate::CodegenCx,
    profiler: &SelfProfilerRef,
    cached_context: &mut Context,
    module: &mut dyn Module,
    codegened_func: CodegenedFunction,
) {
    let _timer =
        profiler.generic_activity_with_arg("compile function", &*codegened_func.symbol_name);

    let clif_comments = codegened_func.clif_comments;

    // Store function in context
    let context = cached_context;
    context.clear();
    context.func = codegened_func.func;

    #[cfg(any())] // This is never true
    let _clif_guard = {
        use std::fmt::Write;

        let func_clone = context.func.clone();
        let clif_comments_clone = clif_comments.clone();
        let mut clif = String::new();
        for flag in module.isa().flags().iter() {
            writeln!(clif, "set {}", flag).unwrap();
        }
        write!(clif, "target {}", module.isa().triple().architecture.to_string()).unwrap();
        for isa_flag in module.isa().isa_flags().iter() {
            write!(clif, " {}", isa_flag).unwrap();
        }
        writeln!(clif, "\n").unwrap();
        writeln!(clif, "; symbol {}", codegened_func.symbol_name).unwrap();
        crate::PrintOnPanic(move || {
            let mut clif = clif.clone();
            ::cranelift_codegen::write::decorate_function(
                &mut &clif_comments_clone,
                &mut clif,
                &func_clone,
            )
            .unwrap();
            clif
        })
    };

    // Define function
    profiler.generic_activity("define function").run(|| {
        context.want_disasm = cx.should_write_ir;
        match module.define_function(codegened_func.func_id, context) {
            Ok(()) => {}
            Err(ModuleError::Compilation(CodegenError::ImplLimitExceeded)) => {
                let early_dcx = rustc_session::EarlyDiagCtxt::new(
                    rustc_session::config::ErrorOutputType::default(),
                );
                early_dcx.early_fatal(format!(
                    "backend implementation limit exceeded while compiling {name}",
                    name = codegened_func.symbol_name
                ));
            }
            Err(ModuleError::Compilation(CodegenError::Verifier(err))) => {
                let early_dcx = rustc_session::EarlyDiagCtxt::new(
                    rustc_session::config::ErrorOutputType::default(),
                );
                let _ = early_dcx.early_err(format!("{:?}", err));
                let pretty_error = cranelift_codegen::print_errors::pretty_verifier_error(
                    &context.func,
                    Some(Box::new(&clif_comments)),
                    err,
                );
                early_dcx.early_fatal(format!("cranelift verify error:\n{}", pretty_error));
            }
            Err(err) => {
                panic!("Error while defining {name}: {err:?}", name = codegened_func.symbol_name);
            }
        }
    });

    if cx.should_write_ir {
        // Write optimized function to file for debugging
        crate::pretty_clif::write_clif_file(
            &cx.output_filenames,
            &codegened_func.symbol_name,
            "opt",
            module.isa(),
            &context.func,
            &clif_comments,
        );

        if let Some(disasm) = &context.compiled_code().unwrap().vcode {
            crate::pretty_clif::write_ir_file(
                &cx.output_filenames,
                &format!("{}.vcode", codegened_func.symbol_name),
                |file| file.write_all(disasm.as_bytes()),
            )
        }
    }

    // Define debuginfo for function
    let debug_context = &mut cx.debug_context;
    profiler.generic_activity("generate debug info").run(|| {
        if let Some(debug_context) = debug_context {
            codegened_func.func_debug_cx.unwrap().finalize(
                debug_context,
                codegened_func.func_id,
                context,
            );
        }
    });
}

fn verify_func(tcx: TyCtxt<'_>, writer: &crate::pretty_clif::CommentWriter, func: &Function) {
    if !enable_verifier(tcx.sess) {
        return;
    }

    tcx.prof.generic_activity("verify clif ir").run(|| {
        let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        match cranelift_codegen::verify_function(&func, &flags) {
            Ok(_) => {}
            Err(err) => {
                tcx.dcx().err(format!("{:?}", err));
                let pretty_error = cranelift_codegen::print_errors::pretty_verifier_error(
                    &func,
                    Some(Box::new(writer)),
                    err,
                );
                tcx.dcx().fatal(format!("cranelift verify error:\n{}", pretty_error));
            }
        }
    });
}

fn codegen_fn_body(fx: &mut FunctionCx<'_, '_, '_>, start_block: Block) {
    let arg_uninhabited = fx
        .mir
        .args_iter()
        .any(|arg| fx.layout_of(fx.monomorphize(fx.mir.local_decls[arg].ty)).is_uninhabited());
    if arg_uninhabited {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
        return;
    }
    fx.tcx
        .prof
        .generic_activity("codegen prelude")
        .run(|| crate::abi::codegen_fn_prelude(fx, start_block));

    let reachable_blocks = traversal::mono_reachable_as_bitset(fx.mir, fx.tcx, fx.instance);

    for (bb, bb_data) in fx.mir.basic_blocks.iter_enumerated() {
        let block = fx.get_block(bb);
        fx.bcx.switch_to_block(block);

        if !reachable_blocks.contains(bb) {
            // We want to skip this block, because it's not reachable. But we still create
            // the block so terminators in other blocks can reference it.
            fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
            continue;
        }

        if bb_data.is_cleanup {
            // Unwinding after panicking is not supported
            continue;

            // FIXME Once unwinding is supported and Cranelift supports marking blocks as cold, do
            // so for cleanup blocks.
        }

        fx.bcx.ins().nop();
        for stmt in &bb_data.statements {
            fx.set_debug_loc(stmt.source_info);
            codegen_stmt(fx, block, stmt);
        }

        if fx.clif_comments.enabled() {
            let mut terminator_head = "\n".to_string();
            with_no_trimmed_paths!({
                bb_data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
            });
            let inst = fx.bcx.func.layout.last_inst(block).unwrap();
            fx.add_post_comment(inst, terminator_head);
        }

        let source_info = bb_data.terminator().source_info;
        fx.set_debug_loc(source_info);

        let _print_guard =
            crate::PrintOnPanic(|| format!("terminator {:?}", bb_data.terminator().kind));

        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                if let TerminatorKind::Return = fx.mir[*target].terminator().kind {
                    let mut can_immediately_return = true;
                    for stmt in &fx.mir[*target].statements {
                        if let StatementKind::StorageDead(_) = stmt.kind {
                        } else {
                            // FIXME Can sometimes happen, see rust-lang/rust#70531
                            can_immediately_return = false;
                            break;
                        }
                    }

                    if can_immediately_return {
                        crate::abi::codegen_return(fx);
                        continue;
                    }
                }

                let block = fx.get_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            TerminatorKind::Return => {
                crate::abi::codegen_return(fx);
            }
            TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                if !fx.tcx.sess.overflow_checks() && msg.is_optional_overflow_check() {
                    let target = fx.get_block(*target);
                    fx.bcx.ins().jump(target, &[]);
                    continue;
                }
                let cond = codegen_operand(fx, cond).load_scalar(fx);

                let target = fx.get_block(*target);
                let failure = fx.bcx.create_block();

                if *expected {
                    fx.bcx.ins().brif(cond, target, &[], failure, &[]);
                } else {
                    fx.bcx.ins().brif(cond, failure, &[], target, &[]);
                };

                fx.bcx.switch_to_block(failure);
                fx.bcx.ins().nop();

                match &**msg {
                    AssertKind::BoundsCheck { ref len, ref index } => {
                        let len = codegen_operand(fx, len).load_scalar(fx);
                        let index = codegen_operand(fx, index).load_scalar(fx);
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicBoundsCheck,
                            &[index, len, location],
                            *unwind,
                            source_info.span,
                        );
                    }
                    AssertKind::MisalignedPointerDereference { ref required, ref found } => {
                        let required = codegen_operand(fx, required).load_scalar(fx);
                        let found = codegen_operand(fx, found).load_scalar(fx);
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicMisalignedPointerDereference,
                            &[required, found, location],
                            *unwind,
                            source_info.span,
                        );
                    }
                    AssertKind::NullPointerDereference => {
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicNullPointerDereference,
                            &[location],
                            *unwind,
                            source_info.span,
                        )
                    }
                    AssertKind::InvalidEnumConstruction(source) => {
                        let source = codegen_operand(fx, source).load_scalar(fx);
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicInvalidEnumConstruction,
                            &[source, location],
                            *unwind,
                            source_info.span,
                        )
                    }
                    _ => {
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            msg.panic_function(),
                            &[location],
                            *unwind,
                            source_info.span,
                        );
                    }
                }
            }

            TerminatorKind::SwitchInt { discr, targets } => {
                let discr = codegen_operand(fx, discr);
                let switch_ty = discr.layout().ty;
                let discr = discr.load_scalar(fx);

                let use_bool_opt = switch_ty.kind() == fx.tcx.types.bool.kind()
                    || (targets.iter().count() == 1 && targets.iter().next().unwrap().0 == 0);
                if use_bool_opt {
                    assert_eq!(targets.iter().count(), 1);
                    let (then_value, then_block) = targets.iter().next().unwrap();
                    let then_block = fx.get_block(then_block);
                    let else_block = fx.get_block(targets.otherwise());
                    let test_zero = match then_value {
                        0 => true,
                        1 => false,
                        _ => unreachable!("{:?}", targets),
                    };

                    let (discr, is_inverted) =
                        crate::optimize::peephole::maybe_unwrap_bool_not(&mut fx.bcx, discr);
                    let test_zero = if is_inverted { !test_zero } else { test_zero };
                    if let Some(taken) = crate::optimize::peephole::maybe_known_branch_taken(
                        &fx.bcx, discr, test_zero,
                    ) {
                        if taken {
                            fx.bcx.ins().jump(then_block, &[]);
                        } else {
                            fx.bcx.ins().jump(else_block, &[]);
                        }
                    } else {
                        if test_zero {
                            fx.bcx.ins().brif(discr, else_block, &[], then_block, &[]);
                        } else {
                            fx.bcx.ins().brif(discr, then_block, &[], else_block, &[]);
                        }
                    }
                } else {
                    let mut switch = ::cranelift_frontend::Switch::new();
                    for (value, block) in targets.iter() {
                        let block = fx.get_block(block);
                        switch.set_entry(value, block);
                    }
                    let otherwise_block = fx.get_block(targets.otherwise());
                    switch.emit(&mut fx.bcx, discr, otherwise_block);
                }
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                fn_span,
                unwind,
                call_source: _,
            } => {
                fx.tcx.prof.generic_activity("codegen call").run(|| {
                    crate::abi::codegen_terminator_call(
                        fx,
                        mir::SourceInfo { span: *fn_span, ..source_info },
                        func,
                        args,
                        *destination,
                        *target,
                        *unwind,
                    )
                });
            }
            // FIXME(explicit_tail_calls): add support for tail calls to the cranelift backend, once cranelift supports tail calls
            TerminatorKind::TailCall { fn_span, .. } => span_bug!(
                *fn_span,
                "tail calls are not yet supported in `rustc_codegen_cranelift` backend"
            ),
            TerminatorKind::InlineAsm {
                asm_macro: _,
                template,
                operands,
                options,
                targets,
                line_spans: _,
                unwind: _,
            } => {
                if options.contains(InlineAsmOptions::MAY_UNWIND) {
                    fx.tcx.dcx().span_fatal(
                        source_info.span,
                        "cranelift doesn't support unwinding from inline assembly.",
                    );
                }

                let have_labels = if options.contains(InlineAsmOptions::NORETURN) {
                    !targets.is_empty()
                } else {
                    targets.len() > 1
                };
                if have_labels {
                    fx.tcx.dcx().span_fatal(
                        source_info.span,
                        "cranelift doesn't support labels in inline assembly.",
                    );
                }

                crate::inline_asm::codegen_inline_asm_terminator(
                    fx,
                    source_info.span,
                    template,
                    operands,
                    *options,
                    targets.get(0).copied(),
                );
            }
            TerminatorKind::UnwindTerminate(reason) => {
                codegen_unwind_terminate(fx, source_info.span, *reason);
            }
            TerminatorKind::UnwindResume => {
                // FIXME implement unwinding
                fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
            }
            TerminatorKind::Unreachable => {
                fx.bcx.set_cold_block(block);
                fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
            }
            TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop => {
                bug!("shouldn't exist at codegen {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut } => {
                assert!(
                    async_fut.is_none() && drop.is_none(),
                    "Async Drop must be expanded or reset to sync before codegen"
                );
                let drop_place = codegen_place(fx, *place);
                crate::abi::codegen_drop(fx, source_info, drop_place, *target, *unwind);
            }
        };
    }
}

fn codegen_stmt<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, cur_block: Block, stmt: &Statement<'tcx>) {
    let _print_guard = crate::PrintOnPanic(|| format!("stmt {:?}", stmt));

    fx.set_debug_loc(stmt.source_info);

    match &stmt.kind {
        StatementKind::StorageLive(..) | StatementKind::StorageDead(..) => {} // Those are not very useful
        _ => {
            if fx.clif_comments.enabled() {
                let inst = fx.bcx.func.layout.last_inst(cur_block).unwrap();
                with_no_trimmed_paths!({
                    fx.add_post_comment(inst, format!("{:?}", stmt));
                });
            }
        }
    }

    match &stmt.kind {
        StatementKind::SetDiscriminant { place, variant_index } => {
            let place = codegen_place(fx, **place);
            crate::discriminant::codegen_set_discriminant(fx, place, *variant_index);
        }
        StatementKind::Assign(to_place_and_rval) => {
            let lval = codegen_place(fx, to_place_and_rval.0);
            let dest_layout = lval.layout();
            match to_place_and_rval.1 {
                Rvalue::Use(ref operand) => {
                    let val = codegen_operand(fx, operand);
                    lval.write_cvalue(fx, val);
                }
                Rvalue::CopyForDeref(place) => {
                    let cplace = codegen_place(fx, place);
                    let val = cplace.to_cvalue(fx);
                    lval.write_cvalue(fx, val)
                }
                Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
                    let place = codegen_place(fx, place);
                    let ref_ = place.place_ref(fx, lval.layout());
                    lval.write_cvalue(fx, ref_);
                }
                Rvalue::ThreadLocalRef(def_id) => {
                    let val = crate::constant::codegen_tls_ref(fx, def_id, lval.layout());
                    lval.write_cvalue(fx, val);
                }
                Rvalue::BinaryOp(bin_op, ref lhs_rhs) => {
                    let lhs = codegen_operand(fx, &lhs_rhs.0);
                    let rhs = codegen_operand(fx, &lhs_rhs.1);

                    let res = if let Some(bin_op) = bin_op.overflowing_to_wrapping() {
                        crate::num::codegen_checked_int_binop(fx, bin_op, lhs, rhs)
                    } else {
                        crate::num::codegen_binop(fx, bin_op, lhs, rhs)
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::UnaryOp(un_op, ref operand) => {
                    let operand = codegen_operand(fx, operand);
                    let layout = operand.layout();
                    let res = match un_op {
                        UnOp::Not => {
                            let val = operand.load_scalar(fx);
                            match layout.ty.kind() {
                                ty::Bool => {
                                    let res = fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0);
                                    CValue::by_val(res, layout)
                                }
                                ty::Uint(_) | ty::Int(_) => {
                                    CValue::by_val(fx.bcx.ins().bnot(val), layout)
                                }
                                _ => unreachable!("un op Not for {:?}", layout.ty),
                            }
                        }
                        UnOp::Neg => {
                            let val = operand.load_scalar(fx);
                            match layout.ty.kind() {
                                ty::Int(_) => CValue::by_val(fx.bcx.ins().ineg(val), layout),
                                // FIXME(bytecodealliance/wasmtime#8312): Remove
                                // once backend lowerings have been added to
                                // Cranelift.
                                ty::Float(FloatTy::F16) => {
                                    CValue::by_val(codegen_f16_f128::neg_f16(fx, val), layout)
                                }
                                ty::Float(FloatTy::F128) => {
                                    CValue::by_val(codegen_f16_f128::neg_f128(fx, val), layout)
                                }
                                ty::Float(_) => CValue::by_val(fx.bcx.ins().fneg(val), layout),
                                _ => unreachable!("un op Neg for {:?}", layout.ty),
                            }
                        }
                        UnOp::PtrMetadata => match layout.backend_repr {
                            BackendRepr::Scalar(_) => CValue::zst(dest_layout),
                            BackendRepr::ScalarPair(_, _) => {
                                CValue::by_val(operand.load_scalar_pair(fx).1, dest_layout)
                            }
                            _ => bug!("Unexpected `PtrToMetadata` operand: {operand:?}"),
                        },
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _),
                    ref operand,
                    to_ty,
                ) => {
                    let from_ty = fx.monomorphize(operand.ty(&fx.mir.local_decls, fx.tcx));
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    match *from_ty.kind() {
                        ty::FnDef(def_id, args) => {
                            let func_ref = fx.get_function_ref(
                                Instance::resolve_for_fn_ptr(
                                    fx.tcx,
                                    ty::TypingEnv::fully_monomorphized(),
                                    def_id,
                                    args,
                                )
                                .unwrap(),
                            );
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, to_layout));
                        }
                        _ => bug!("Trying to ReifyFnPointer on non FnDef {:?}", from_ty),
                    }
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer, _),
                    ref operand,
                    to_ty,
                ) => {
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    let operand = codegen_operand(fx, operand);
                    lval.write_cvalue(fx, operand.cast_pointer_to(to_layout));
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(
                        PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
                        _,
                    ),
                    ..,
                ) => {
                    bug!(
                        "{:?} is for borrowck, and should never appear in codegen",
                        to_place_and_rval.1
                    );
                }
                Rvalue::Cast(
                    CastKind::IntToInt
                    | CastKind::FloatToFloat
                    | CastKind::FloatToInt
                    | CastKind::IntToFloat
                    | CastKind::FnPtrToPtr
                    | CastKind::PtrToPtr
                    | CastKind::PointerExposeProvenance
                    | CastKind::PointerWithExposedProvenance,
                    ref operand,
                    to_ty,
                ) => {
                    let operand = codegen_operand(fx, operand);
                    let from_ty = operand.layout().ty;
                    let to_ty = fx.monomorphize(to_ty);

                    fn is_wide_ptr<'tcx>(fx: &FunctionCx<'_, '_, 'tcx>, ty: Ty<'tcx>) -> bool {
                        ty.builtin_deref(true).is_some_and(|pointee_ty| {
                            fx.tcx
                                .type_has_metadata(pointee_ty, ty::TypingEnv::fully_monomorphized())
                        })
                    }

                    if is_wide_ptr(fx, from_ty) {
                        if is_wide_ptr(fx, to_ty) {
                            // wide-ptr -> wide-ptr
                            lval.write_cvalue(fx, operand.cast_pointer_to(dest_layout));
                        } else {
                            // wide-ptr -> thin-ptr
                            let (ptr, _extra) = operand.load_scalar_pair(fx);
                            lval.write_cvalue(fx, CValue::by_val(ptr, dest_layout))
                        }
                    } else {
                        let to_clif_ty = fx.clif_type(to_ty).unwrap();
                        let from = operand.load_scalar(fx);

                        let res = clif_int_or_float_cast(
                            fx,
                            from,
                            type_sign(from_ty),
                            to_clif_ty,
                            type_sign(to_ty),
                        );
                        lval.write_cvalue(fx, CValue::by_val(res, dest_layout));
                    }
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _),
                    ref operand,
                    _to_ty,
                ) => {
                    let operand = codegen_operand(fx, operand);
                    match *operand.layout().ty.kind() {
                        ty::Closure(def_id, args) => {
                            let instance = Instance::resolve_closure(
                                fx.tcx,
                                def_id,
                                args,
                                ty::ClosureKind::FnOnce,
                            );
                            let func_ref = fx.get_function_ref(instance);
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, lval.layout()));
                        }
                        _ => bug!("{} cannot be cast to a fn ptr", operand.layout().ty),
                    }
                }
                Rvalue::Cast(
                    CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                    ref operand,
                    _to_ty,
                ) => {
                    let operand = codegen_operand(fx, operand);
                    crate::unsize::coerce_unsized_into(fx, operand, lval);
                }
                Rvalue::Cast(CastKind::Transmute | CastKind::Subtype, ref operand, _to_ty) => {
                    let operand = codegen_operand(fx, operand);
                    lval.write_cvalue_transmute(fx, operand);
                }
                Rvalue::Discriminant(place) => {
                    let place = codegen_place(fx, place);
                    let value = place.to_cvalue(fx);
                    crate::discriminant::codegen_get_discriminant(fx, lval, value, dest_layout);
                }
                Rvalue::Repeat(ref operand, times) => {
                    let operand = codegen_operand(fx, operand);
                    let times = fx
                        .monomorphize(times)
                        .try_to_target_usize(fx.tcx)
                        .expect("expected monomorphic const in codegen");
                    if operand.layout().size.bytes() == 0 {
                        // Do nothing for ZST's
                    } else if fx.clif_type(operand.layout().ty) == Some(types::I8) {
                        let times = fx.bcx.ins().iconst(fx.pointer_type, times as i64);
                        // FIXME use emit_small_memset where possible
                        let addr = lval.to_ptr().get_addr(fx);
                        let val = operand.load_scalar(fx);
                        fx.bcx.call_memset(fx.target_config, addr, val, times);
                    } else {
                        let loop_block = fx.bcx.create_block();
                        let loop_block2 = fx.bcx.create_block();
                        let done_block = fx.bcx.create_block();
                        let index = fx.bcx.append_block_param(loop_block, fx.pointer_type);
                        let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
                        fx.bcx.ins().jump(loop_block, &[zero.into()]);

                        fx.bcx.switch_to_block(loop_block);
                        let done = fx.bcx.ins().icmp_imm(IntCC::Equal, index, times as i64);
                        fx.bcx.ins().brif(done, done_block, &[], loop_block2, &[]);

                        fx.bcx.switch_to_block(loop_block2);
                        let to = lval.place_index(fx, index);
                        to.write_cvalue(fx, operand);
                        let index = fx.bcx.ins().iadd_imm(index, 1);
                        fx.bcx.ins().jump(loop_block, &[index.into()]);

                        fx.bcx.switch_to_block(done_block);
                        fx.bcx.ins().nop();
                    }
                }
                Rvalue::ShallowInitBox(ref operand, content_ty) => {
                    let content_ty = fx.monomorphize(content_ty);
                    let box_layout = fx.layout_of(Ty::new_box(fx.tcx, content_ty));
                    let operand = codegen_operand(fx, operand);
                    let operand = operand.load_scalar(fx);
                    lval.write_cvalue(fx, CValue::by_val(operand, box_layout));
                }
                Rvalue::NullaryOp(ref null_op, ty) => {
                    assert!(lval.layout().ty.is_sized(fx.tcx, fx.typing_env()));
                    let layout = fx.layout_of(fx.monomorphize(ty));
                    let val = match null_op {
                        NullOp::SizeOf => layout.size.bytes(),
                        NullOp::AlignOf => layout.align.bytes(),
                        NullOp::OffsetOf(fields) => fx
                            .tcx
                            .offset_of_subfield(
                                ty::TypingEnv::fully_monomorphized(),
                                layout,
                                fields.iter(),
                            )
                            .bytes(),
                        NullOp::UbChecks => {
                            let val = fx.tcx.sess.ub_checks();
                            let val = CValue::by_val(
                                fx.bcx.ins().iconst(types::I8, i64::from(val)),
                                fx.layout_of(fx.tcx.types.bool),
                            );
                            lval.write_cvalue(fx, val);
                            return;
                        }
                        NullOp::ContractChecks => {
                            let val = fx.tcx.sess.contract_checks();
                            let val = CValue::by_val(
                                fx.bcx.ins().iconst(types::I8, i64::from(val)),
                                fx.layout_of(fx.tcx.types.bool),
                            );
                            lval.write_cvalue(fx, val);
                            return;
                        }
                    };
                    let val = CValue::by_val(
                        fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(val).unwrap()),
                        fx.layout_of(fx.tcx.types.usize),
                    );
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Aggregate(ref kind, ref operands)
                    if matches!(**kind, AggregateKind::RawPtr(..)) =>
                {
                    let ty = to_place_and_rval.1.ty(&fx.mir.local_decls, fx.tcx);
                    let layout = fx.layout_of(fx.monomorphize(ty));
                    let [data, meta] = &*operands.raw else {
                        bug!("RawPtr fields: {operands:?}");
                    };
                    let data = codegen_operand(fx, data);
                    let meta = codegen_operand(fx, meta);
                    assert!(data.layout().ty.is_raw_ptr());
                    assert!(layout.ty.is_raw_ptr());
                    let ptr_val = if meta.layout().is_zst() {
                        data.cast_pointer_to(layout)
                    } else {
                        CValue::by_val_pair(data.load_scalar(fx), meta.load_scalar(fx), layout)
                    };
                    lval.write_cvalue(fx, ptr_val);
                }
                Rvalue::Aggregate(ref kind, ref operands) => {
                    let (variant_index, variant_dest, active_field_index) = match **kind {
                        mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) => {
                            let variant_dest = lval.downcast_variant(fx, variant_index);
                            (variant_index, variant_dest, active_field_index)
                        }
                        _ => (FIRST_VARIANT, lval, None),
                    };
                    if active_field_index.is_some() {
                        assert_eq!(operands.len(), 1);
                    }
                    for (i, operand) in operands.iter_enumerated() {
                        let operand = codegen_operand(fx, operand);
                        let field_index = active_field_index.unwrap_or(i);
                        let to = if let mir::AggregateKind::Array(_) = **kind {
                            let array_index = i64::from(field_index.as_u32());
                            let index = fx.bcx.ins().iconst(fx.pointer_type, array_index);
                            variant_dest.place_index(fx, index)
                        } else {
                            variant_dest.place_field(fx, field_index)
                        };
                        to.write_cvalue(fx, operand);
                    }
                    crate::discriminant::codegen_set_discriminant(fx, lval, variant_index);
                }
                Rvalue::WrapUnsafeBinder(ref operand, _to_ty) => {
                    let operand = codegen_operand(fx, operand);
                    lval.write_cvalue_transmute(fx, operand);
                }
            }
        }
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Deinit(_)
        | StatementKind::ConstEvalCounter
        | StatementKind::Nop
        | StatementKind::FakeRead(..)
        | StatementKind::Retag { .. }
        | StatementKind::PlaceMention(..)
        | StatementKind::BackwardIncompatibleDropHint { .. }
        | StatementKind::AscribeUserType(..) => {}

        StatementKind::Coverage { .. } => unreachable!(),
        StatementKind::Intrinsic(ref intrinsic) => match &**intrinsic {
            // We ignore `assume` intrinsics, they are only useful for optimizations
            NonDivergingIntrinsic::Assume(_) => {}
            NonDivergingIntrinsic::CopyNonOverlapping(mir::CopyNonOverlapping {
                src,
                dst,
                count,
            }) => {
                let dst = codegen_operand(fx, dst);
                let pointee = dst
                    .layout()
                    .pointee_info_at(fx, rustc_abi::Size::ZERO)
                    .expect("Expected pointer");
                let dst = dst.load_scalar(fx);
                let src = codegen_operand(fx, src).load_scalar(fx);
                let count = codegen_operand(fx, count).load_scalar(fx);
                let elem_size: u64 = pointee.size.bytes();
                let bytes = if elem_size != 1 {
                    fx.bcx.ins().imul_imm(count, elem_size as i64)
                } else {
                    count
                };
                fx.bcx.call_memcpy(fx.target_config, dst, src, bytes);
            }
        },
    }
}

fn codegen_array_len<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, place: CPlace<'tcx>) -> Value {
    match *place.layout().ty.kind() {
        ty::Array(_elem_ty, len) => {
            let len = fx
                .monomorphize(len)
                .try_to_target_usize(fx.tcx)
                .expect("expected monomorphic const in codegen") as i64;
            fx.bcx.ins().iconst(fx.pointer_type, len)
        }
        ty::Slice(_elem_ty) => place.to_ptr_unsized().1,
        _ => bug!("Rvalue::Len({:?})", place),
    }
}

pub(crate) fn codegen_place<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    place: Place<'tcx>,
) -> CPlace<'tcx> {
    let mut cplace = fx.get_local_place(place.local);

    for elem in place.projection {
        match elem {
            PlaceElem::Deref => {
                cplace = cplace.place_deref(fx);
            }
            PlaceElem::OpaqueCast(ty) => bug!("encountered OpaqueCast({ty}) in codegen"),
            PlaceElem::UnwrapUnsafeBinder(ty) => {
                cplace = cplace.place_transmute_type(fx, fx.monomorphize(ty));
            }
            PlaceElem::Field(field, _ty) => {
                cplace = cplace.place_field(fx, field);
            }
            PlaceElem::Index(local) => {
                let index = fx.get_local_place(local).to_cvalue(fx).load_scalar(fx);
                cplace = cplace.place_index(fx, index);
            }
            PlaceElem::ConstantIndex { offset, min_length: _, from_end } => {
                let offset: u64 = offset;
                let index = if !from_end {
                    fx.bcx.ins().iconst(fx.pointer_type, offset as i64)
                } else {
                    let len = codegen_array_len(fx, cplace);
                    fx.bcx.ins().iadd_imm(len, -(offset as i64))
                };
                cplace = cplace.place_index(fx, index);
            }
            PlaceElem::Subslice { from, to, from_end } => {
                // These indices are generated by slice patterns.
                // slice[from:-to] in Python terms.

                let from: u64 = from;
                let to: u64 = to;

                match cplace.layout().ty.kind() {
                    ty::Array(elem_ty, _len) => {
                        assert!(!from_end, "array subslices are never `from_end`");
                        let elem_layout = fx.layout_of(*elem_ty);
                        let ptr = cplace.to_ptr();
                        cplace = CPlace::for_ptr(
                            ptr.offset_i64(fx, elem_layout.size.bytes() as i64 * (from as i64)),
                            fx.layout_of(Ty::new_array(fx.tcx, *elem_ty, to - from)),
                        );
                    }
                    ty::Slice(elem_ty) => {
                        assert!(from_end, "slice subslices should be `from_end`");
                        let elem_layout = fx.layout_of(*elem_ty);
                        let (ptr, len) = cplace.to_ptr_unsized();
                        cplace = CPlace::for_ptr_with_extra(
                            ptr.offset_i64(fx, elem_layout.size.bytes() as i64 * (from as i64)),
                            fx.bcx.ins().iadd_imm(len, -(from as i64 + to as i64)),
                            cplace.layout(),
                        );
                    }
                    _ => unreachable!(),
                }
            }
            PlaceElem::Downcast(_adt_def, variant) => {
                cplace = cplace.downcast_variant(fx, variant);
            }
        }
    }

    cplace
}

pub(crate) fn codegen_operand<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    operand: &Operand<'tcx>,
) -> CValue<'tcx> {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => {
            let cplace = codegen_place(fx, *place);
            cplace.to_cvalue(fx)
        }
        Operand::Constant(const_) => crate::constant::codegen_constant_operand(fx, const_),
    }
}

pub(crate) fn codegen_panic_nounwind<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    msg_str: &str,
    span: Span,
) {
    let msg_ptr = fx.anonymous_str(msg_str);
    let msg_len = fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(msg_str.len()).unwrap());
    let args = [msg_ptr, msg_len];

    codegen_panic_inner(
        fx,
        rustc_hir::LangItem::PanicNounwind,
        &args,
        UnwindAction::Terminate(UnwindTerminateReason::Abi),
        span,
    );
}

pub(crate) fn codegen_unwind_terminate<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    span: Span,
    reason: UnwindTerminateReason,
) {
    codegen_panic_inner(fx, reason.lang_item(), &[], UnwindAction::Unreachable, span);
}

fn codegen_panic_inner<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    lang_item: rustc_hir::LangItem,
    args: &[Value],
    _unwind: UnwindAction,
    span: Span,
) {
    fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());

    let def_id = fx.tcx.require_lang_item(lang_item, span);

    let instance = Instance::mono(fx.tcx, def_id);

    if is_call_from_compiler_builtins_to_upstream_monomorphization(fx.tcx, instance) {
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return;
    }

    let symbol_name = fx.tcx.symbol_name(instance).name;

    // FIXME implement cleanup on exceptions

    fx.lib_call(
        symbol_name,
        args.iter().map(|&arg| AbiParam::new(fx.bcx.func.dfg.value_type(arg))).collect(),
        vec![],
        args,
    );

    fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
}
