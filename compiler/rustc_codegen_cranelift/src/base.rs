//! Codegen of a single function

use rustc_ast::InlineAsmOptions;
use rustc_index::vec::IndexVec;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::print::with_no_trimmed_paths;

use crate::constant::ConstantCx;
use crate::debuginfo::FunctionDebugContext;
use crate::prelude::*;
use crate::pretty_clif::CommentWriter;

pub(crate) struct CodegenedFunction {
    symbol_name: String,
    func_id: FuncId,
    func: Function,
    clif_comments: CommentWriter,
    func_debug_cx: Option<FunctionDebugContext>,
}

#[cfg_attr(not(feature = "jit"), allow(dead_code))]
pub(crate) fn codegen_and_compile_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut crate::CodegenCx,
    cached_context: &mut Context,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) {
    let _inst_guard =
        crate::PrintOnPanic(|| format!("{:?} {}", instance, tcx.symbol_name(instance).name));

    let cached_func = std::mem::replace(&mut cached_context.func, Function::new());
    let codegened_func = codegen_fn(tcx, cx, cached_func, module, instance);

    compile_fn(cx, cached_context, module, codegened_func);
}

pub(crate) fn codegen_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut crate::CodegenCx,
    cached_func: Function,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) -> CodegenedFunction {
    debug_assert!(!instance.substs.needs_infer());

    let mir = tcx.instance_mir(instance.def);
    let _mir_guard = crate::PrintOnPanic(|| {
        let mut buf = Vec::new();
        with_no_trimmed_paths!({
            rustc_middle::mir::pretty::write_mir_fn(tcx, mir, &mut |_, _| Ok(()), &mut buf)
                .unwrap();
        });
        String::from_utf8_lossy(&buf).into_owned()
    });

    // Declare function
    let symbol_name = tcx.symbol_name(instance).name.to_string();
    let sig = get_function_sig(tcx, module.isa().triple(), instance);
    let func_id = module.declare_function(&symbol_name, Linkage::Local, &sig).unwrap();

    // Make the FunctionBuilder
    let mut func_ctx = FunctionBuilderContext::new();
    let mut func = cached_func;
    func.clear();
    func.name = ExternalName::user(0, func_id.as_u32());
    func.signature = sig;
    func.collect_debug_info();

    let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Predefine blocks
    let start_block = bcx.create_block();
    let block_map: IndexVec<BasicBlock, Block> =
        (0..mir.basic_blocks().len()).map(|_| bcx.create_block()).collect();

    // Make FunctionCx
    let target_config = module.target_config();
    let pointer_type = target_config.pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance);

    let func_debug_cx = if let Some(debug_context) = &mut cx.debug_context {
        Some(debug_context.define_function(tcx, &symbol_name, mir.span))
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
        fn_abi: Some(RevealAllLayoutCx(tcx).fn_abi_of_instance(instance, ty::List::empty())),

        bcx,
        block_map,
        local_map: IndexVec::with_capacity(mir.local_decls.len()),
        caller_location: None, // set by `codegen_fn_prelude`

        clif_comments,
        last_source_file: None,
        next_ssa_var: 0,
    };

    tcx.sess.time("codegen clif ir", || codegen_fn_body(&mut fx, start_block));

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
    cached_context: &mut Context,
    module: &mut dyn Module,
    codegened_func: CodegenedFunction,
) {
    let clif_comments = codegened_func.clif_comments;

    // Store function in context
    let context = cached_context;
    context.clear();
    context.func = codegened_func.func;

    // If the return block is not reachable, then the SSA builder may have inserted an `iconst.i128`
    // instruction, which doesn't have an encoding.
    context.compute_cfg();
    context.compute_domtree();
    context.eliminate_unreachable_code(module.isa()).unwrap();
    context.dce(module.isa()).unwrap();
    // Some Cranelift optimizations expect the domtree to not yet be computed and as such don't
    // invalidate it when it would change.
    context.domtree.clear();

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
    cx.profiler.verbose_generic_activity("define function").run(|| {
        context.want_disasm = cx.should_write_ir;
        module.define_function(codegened_func.func_id, context).unwrap();
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

        if let Some(disasm) = &context.compiled_code().unwrap().disasm {
            crate::pretty_clif::write_ir_file(
                &cx.output_filenames,
                &format!("{}.vcode", codegened_func.symbol_name),
                |file| file.write_all(disasm.as_bytes()),
            )
        }
    }

    // Define debuginfo for function
    let isa = module.isa();
    let debug_context = &mut cx.debug_context;
    let unwind_context = &mut cx.unwind_context;
    cx.profiler.verbose_generic_activity("generate debug info").run(|| {
        if let Some(debug_context) = debug_context {
            codegened_func.func_debug_cx.unwrap().finalize(
                debug_context,
                codegened_func.func_id,
                context,
            );
        }
        unwind_context.add_function(codegened_func.func_id, &context, isa);
    });
}

pub(crate) fn verify_func(
    tcx: TyCtxt<'_>,
    writer: &crate::pretty_clif::CommentWriter,
    func: &Function,
) {
    tcx.sess.time("verify clif ir", || {
        let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        match cranelift_codegen::verify_function(&func, &flags) {
            Ok(_) => {}
            Err(err) => {
                tcx.sess.err(&format!("{:?}", err));
                let pretty_error = cranelift_codegen::print_errors::pretty_verifier_error(
                    &func,
                    Some(Box::new(writer)),
                    err,
                );
                tcx.sess.fatal(&format!("cranelift verify error:\n{}", pretty_error));
            }
        }
    });
}

fn codegen_fn_body(fx: &mut FunctionCx<'_, '_, '_>, start_block: Block) {
    if !crate::constant::check_constants(fx) {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        // compilation should have been aborted
        fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
        return;
    }

    let arg_uninhabited = fx
        .mir
        .args_iter()
        .any(|arg| fx.layout_of(fx.monomorphize(fx.mir.local_decls[arg].ty)).abi.is_uninhabited());
    if arg_uninhabited {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
        return;
    }
    fx.tcx.sess.time("codegen prelude", || crate::abi::codegen_fn_prelude(fx, start_block));

    for (bb, bb_data) in fx.mir.basic_blocks().iter_enumerated() {
        let block = fx.get_block(bb);
        fx.bcx.switch_to_block(block);

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
            fx.add_comment(inst, terminator_head);
        }

        let source_info = bb_data.terminator().source_info;
        fx.set_debug_loc(source_info);

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
            TerminatorKind::Assert { cond, expected, msg, target, cleanup: _ } => {
                if !fx.tcx.sess.overflow_checks() {
                    if let mir::AssertKind::OverflowNeg(_) = *msg {
                        let target = fx.get_block(*target);
                        fx.bcx.ins().jump(target, &[]);
                        continue;
                    }
                }
                let cond = codegen_operand(fx, cond).load_scalar(fx);

                let target = fx.get_block(*target);
                let failure = fx.bcx.create_block();
                fx.bcx.set_cold_block(failure);

                if *expected {
                    fx.bcx.ins().brz(cond, failure, &[]);
                } else {
                    fx.bcx.ins().brnz(cond, failure, &[]);
                };
                fx.bcx.ins().jump(target, &[]);

                fx.bcx.switch_to_block(failure);
                fx.bcx.ins().nop();

                match msg {
                    AssertKind::BoundsCheck { ref len, ref index } => {
                        let len = codegen_operand(fx, len).load_scalar(fx);
                        let index = codegen_operand(fx, index).load_scalar(fx);
                        let location = fx.get_caller_location(source_info).load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicBoundsCheck,
                            &[index, len, location],
                            source_info.span,
                        );
                    }
                    _ => {
                        let msg_str = msg.description();
                        codegen_panic(fx, msg_str, source_info);
                    }
                }
            }

            TerminatorKind::SwitchInt { discr, switch_ty, targets } => {
                let discr = codegen_operand(fx, discr).load_scalar(fx);

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

                    let discr = crate::optimize::peephole::maybe_unwrap_bint(&mut fx.bcx, discr);
                    let (discr, is_inverted) =
                        crate::optimize::peephole::maybe_unwrap_bool_not(&mut fx.bcx, discr);
                    let test_zero = if is_inverted { !test_zero } else { test_zero };
                    let discr = crate::optimize::peephole::maybe_unwrap_bint(&mut fx.bcx, discr);
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
                            fx.bcx.ins().brz(discr, then_block, &[]);
                            fx.bcx.ins().jump(else_block, &[]);
                        } else {
                            fx.bcx.ins().brnz(discr, then_block, &[]);
                            fx.bcx.ins().jump(else_block, &[]);
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
                cleanup: _,
                from_hir_call: _,
            } => {
                fx.tcx.sess.time("codegen call", || {
                    crate::abi::codegen_terminator_call(
                        fx,
                        mir::SourceInfo { span: *fn_span, ..source_info },
                        func,
                        args,
                        *destination,
                        *target,
                    )
                });
            }
            TerminatorKind::InlineAsm {
                template,
                operands,
                options,
                destination,
                line_spans: _,
                cleanup: _,
            } => {
                if options.contains(InlineAsmOptions::MAY_UNWIND) {
                    fx.tcx.sess.span_fatal(
                        source_info.span,
                        "cranelift doesn't support unwinding from inline assembly.",
                    );
                }

                crate::inline_asm::codegen_inline_asm(
                    fx,
                    source_info.span,
                    template,
                    operands,
                    *options,
                    *destination,
                );
            }
            TerminatorKind::Resume | TerminatorKind::Abort => {
                // FIXME implement unwinding
                fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
            }
            TerminatorKind::Unreachable => {
                fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
            }
            TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::GeneratorDrop => {
                bug!("shouldn't exist at codegen {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop { place, target, unwind: _ } => {
                let drop_place = codegen_place(fx, *place);
                crate::abi::codegen_drop(fx, source_info, drop_place);

                let target_block = fx.get_block(*target);
                fx.bcx.ins().jump(target_block, &[]);
            }
        };
    }

    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
}

fn codegen_stmt<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    #[allow(unused_variables)] cur_block: Block,
    stmt: &Statement<'tcx>,
) {
    let _print_guard = crate::PrintOnPanic(|| format!("stmt {:?}", stmt));

    fx.set_debug_loc(stmt.source_info);

    #[cfg(any())] // This is never true
    match &stmt.kind {
        StatementKind::StorageLive(..) | StatementKind::StorageDead(..) => {} // Those are not very useful
        _ => {
            if fx.clif_comments.enabled() {
                let inst = fx.bcx.func.layout.last_inst(cur_block).unwrap();
                fx.add_comment(inst, format!("{:?}", stmt));
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
                Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
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

                    let res = crate::num::codegen_binop(fx, bin_op, lhs, rhs);
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, ref lhs_rhs) => {
                    let lhs = codegen_operand(fx, &lhs_rhs.0);
                    let rhs = codegen_operand(fx, &lhs_rhs.1);

                    let res = if !fx.tcx.sess.overflow_checks() {
                        let val =
                            crate::num::codegen_int_binop(fx, bin_op, lhs, rhs).load_scalar(fx);
                        let is_overflow = fx.bcx.ins().iconst(types::I8, 0);
                        CValue::by_val_pair(val, is_overflow, lval.layout())
                    } else {
                        crate::num::codegen_checked_int_binop(fx, bin_op, lhs, rhs)
                    };

                    lval.write_cvalue(fx, res);
                }
                Rvalue::UnaryOp(un_op, ref operand) => {
                    let operand = codegen_operand(fx, operand);
                    let layout = operand.layout();
                    let val = operand.load_scalar(fx);
                    let res = match un_op {
                        UnOp::Not => match layout.ty.kind() {
                            ty::Bool => {
                                let res = fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0);
                                CValue::by_val(fx.bcx.ins().bint(types::I8, res), layout)
                            }
                            ty::Uint(_) | ty::Int(_) => {
                                CValue::by_val(fx.bcx.ins().bnot(val), layout)
                            }
                            _ => unreachable!("un op Not for {:?}", layout.ty),
                        },
                        UnOp::Neg => match layout.ty.kind() {
                            ty::Int(IntTy::I128) => {
                                // FIXME remove this case once ineg.i128 works
                                let zero =
                                    CValue::const_val(fx, layout, ty::ScalarInt::null(layout.size));
                                crate::num::codegen_int_binop(fx, BinOp::Sub, zero, operand)
                            }
                            ty::Int(_) => CValue::by_val(fx.bcx.ins().ineg(val), layout),
                            ty::Float(_) => CValue::by_val(fx.bcx.ins().fneg(val), layout),
                            _ => unreachable!("un op Neg for {:?}", layout.ty),
                        },
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::Cast(
                    CastKind::Pointer(PointerCast::ReifyFnPointer),
                    ref operand,
                    to_ty,
                ) => {
                    let from_ty = fx.monomorphize(operand.ty(&fx.mir.local_decls, fx.tcx));
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    match *from_ty.kind() {
                        ty::FnDef(def_id, substs) => {
                            let func_ref = fx.get_function_ref(
                                Instance::resolve_for_fn_ptr(
                                    fx.tcx,
                                    ParamEnv::reveal_all(),
                                    def_id,
                                    substs,
                                )
                                .unwrap()
                                .polymorphize(fx.tcx),
                            );
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, to_layout));
                        }
                        _ => bug!("Trying to ReifyFnPointer on non FnDef {:?}", from_ty),
                    }
                }
                Rvalue::Cast(
                    CastKind::Pointer(PointerCast::UnsafeFnPointer),
                    ref operand,
                    to_ty,
                )
                | Rvalue::Cast(
                    CastKind::Pointer(PointerCast::MutToConstPointer),
                    ref operand,
                    to_ty,
                )
                | Rvalue::Cast(
                    CastKind::Pointer(PointerCast::ArrayToPointer),
                    ref operand,
                    to_ty,
                ) => {
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    let operand = codegen_operand(fx, operand);
                    lval.write_cvalue(fx, operand.cast_pointer_to(to_layout));
                }
                Rvalue::Cast(
                    CastKind::Misc
                    | CastKind::PointerExposeAddress
                    | CastKind::PointerFromExposedAddress,
                    ref operand,
                    to_ty,
                ) => {
                    let operand = codegen_operand(fx, operand);
                    let from_ty = operand.layout().ty;
                    let to_ty = fx.monomorphize(to_ty);

                    fn is_fat_ptr<'tcx>(fx: &FunctionCx<'_, '_, 'tcx>, ty: Ty<'tcx>) -> bool {
                        ty.builtin_deref(true)
                            .map(|ty::TypeAndMut { ty: pointee_ty, mutbl: _ }| {
                                has_ptr_meta(fx.tcx, pointee_ty)
                            })
                            .unwrap_or(false)
                    }

                    if is_fat_ptr(fx, from_ty) {
                        if is_fat_ptr(fx, to_ty) {
                            // fat-ptr -> fat-ptr
                            lval.write_cvalue(fx, operand.cast_pointer_to(dest_layout));
                        } else {
                            // fat-ptr -> thin-ptr
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
                    CastKind::Pointer(PointerCast::ClosureFnPointer(_)),
                    ref operand,
                    _to_ty,
                ) => {
                    let operand = codegen_operand(fx, operand);
                    match *operand.layout().ty.kind() {
                        ty::Closure(def_id, substs) => {
                            let instance = Instance::resolve_closure(
                                fx.tcx,
                                def_id,
                                substs,
                                ty::ClosureKind::FnOnce,
                            )
                            .expect("failed to normalize and resolve closure during codegen")
                            .polymorphize(fx.tcx);
                            let func_ref = fx.get_function_ref(instance);
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, lval.layout()));
                        }
                        _ => bug!("{} cannot be cast to a fn ptr", operand.layout().ty),
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), ref operand, _to_ty) => {
                    let operand = codegen_operand(fx, operand);
                    operand.unsize_value(fx, lval);
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
                        .eval(fx.tcx, ParamEnv::reveal_all())
                        .kind()
                        .try_to_bits(fx.tcx.data_layout.pointer_size)
                        .unwrap();
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
                        fx.bcx.ins().jump(loop_block, &[zero]);

                        fx.bcx.switch_to_block(loop_block);
                        let done = fx.bcx.ins().icmp_imm(IntCC::Equal, index, times as i64);
                        fx.bcx.ins().brnz(done, done_block, &[]);
                        fx.bcx.ins().jump(loop_block2, &[]);

                        fx.bcx.switch_to_block(loop_block2);
                        let to = lval.place_index(fx, index);
                        to.write_cvalue(fx, operand);
                        let index = fx.bcx.ins().iadd_imm(index, 1);
                        fx.bcx.ins().jump(loop_block, &[index]);

                        fx.bcx.switch_to_block(done_block);
                        fx.bcx.ins().nop();
                    }
                }
                Rvalue::Len(place) => {
                    let place = codegen_place(fx, place);
                    let usize_layout = fx.layout_of(fx.tcx.types.usize);
                    let len = codegen_array_len(fx, place);
                    lval.write_cvalue(fx, CValue::by_val(len, usize_layout));
                }
                Rvalue::ShallowInitBox(ref operand, content_ty) => {
                    let content_ty = fx.monomorphize(content_ty);
                    let box_layout = fx.layout_of(fx.tcx.mk_box(content_ty));
                    let operand = codegen_operand(fx, operand);
                    let operand = operand.load_scalar(fx);
                    lval.write_cvalue(fx, CValue::by_val(operand, box_layout));
                }
                Rvalue::NullaryOp(null_op, ty) => {
                    assert!(
                        lval.layout()
                            .ty
                            .is_sized(fx.tcx.at(stmt.source_info.span), ParamEnv::reveal_all())
                    );
                    let layout = fx.layout_of(fx.monomorphize(ty));
                    let val = match null_op {
                        NullOp::SizeOf => layout.size.bytes(),
                        NullOp::AlignOf => layout.align.abi.bytes(),
                    };
                    let val = CValue::const_val(fx, fx.layout_of(fx.tcx.types.usize), val.into());
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Aggregate(ref kind, ref operands) => match kind.as_ref() {
                    AggregateKind::Array(_ty) => {
                        for (i, operand) in operands.iter().enumerate() {
                            let operand = codegen_operand(fx, operand);
                            let index = fx.bcx.ins().iconst(fx.pointer_type, i as i64);
                            let to = lval.place_index(fx, index);
                            to.write_cvalue(fx, operand);
                        }
                    }
                    _ => unreachable!("shouldn't exist at codegen {:?}", to_place_and_rval.1),
                },
            }
        }
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Deinit(_)
        | StatementKind::Nop
        | StatementKind::FakeRead(..)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..) => {}

        StatementKind::Coverage { .. } => fx.tcx.sess.fatal("-Zcoverage is unimplemented"),
        StatementKind::CopyNonOverlapping(inner) => {
            let dst = codegen_operand(fx, &inner.dst);
            let pointee = dst
                .layout()
                .pointee_info_at(fx, rustc_target::abi::Size::ZERO)
                .expect("Expected pointer");
            let dst = dst.load_scalar(fx);
            let src = codegen_operand(fx, &inner.src).load_scalar(fx);
            let count = codegen_operand(fx, &inner.count).load_scalar(fx);
            let elem_size: u64 = pointee.size.bytes();
            let bytes =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };
            fx.bcx.call_memcpy(fx.target_config, dst, src, bytes);
        }
    }
}

fn codegen_array_len<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, place: CPlace<'tcx>) -> Value {
    match *place.layout().ty.kind() {
        ty::Array(_elem_ty, len) => {
            let len = fx.monomorphize(len).eval_usize(fx.tcx, ParamEnv::reveal_all()) as i64;
            fx.bcx.ins().iconst(fx.pointer_type, len)
        }
        ty::Slice(_elem_ty) => {
            place.to_ptr_maybe_unsized().1.expect("Length metadata for slice place")
        }
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
                            fx.layout_of(fx.tcx.mk_array(*elem_ty, to - from)),
                        );
                    }
                    ty::Slice(elem_ty) => {
                        assert!(from_end, "slice subslices should be `from_end`");
                        let elem_layout = fx.layout_of(*elem_ty);
                        let (ptr, len) = cplace.to_ptr_maybe_unsized();
                        let len = len.unwrap();
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
        Operand::Constant(const_) => crate::constant::codegen_constant(fx, const_),
    }
}

pub(crate) fn codegen_panic<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    msg_str: &str,
    source_info: mir::SourceInfo,
) {
    let location = fx.get_caller_location(source_info).load_scalar(fx);

    let msg_ptr = fx.anonymous_str(msg_str);
    let msg_len = fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(msg_str.len()).unwrap());
    let args = [msg_ptr, msg_len, location];

    codegen_panic_inner(fx, rustc_hir::LangItem::Panic, &args, source_info.span);
}

pub(crate) fn codegen_panic_inner<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    lang_item: rustc_hir::LangItem,
    args: &[Value],
    span: Span,
) {
    let def_id =
        fx.tcx.lang_items().require(lang_item).unwrap_or_else(|s| fx.tcx.sess.span_fatal(span, &s));

    let instance = Instance::mono(fx.tcx, def_id).polymorphize(fx.tcx);
    let symbol_name = fx.tcx.symbol_name(instance).name;

    fx.lib_call(
        &*symbol_name,
        vec![
            AbiParam::new(fx.pointer_type),
            AbiParam::new(fx.pointer_type),
            AbiParam::new(fx.pointer_type),
        ],
        vec![],
        args,
    );

    fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
}
