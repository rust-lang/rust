//! Codegen of a single function

use cranelift_codegen::binemit::{NullStackMapSink, NullTrapSink};
use rustc_index::vec::IndexVec;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::layout::FnAbiExt;
use rustc_target::abi::call::FnAbi;

use crate::constant::ConstantCx;
use crate::prelude::*;

pub(crate) fn codegen_fn<'tcx>(
    cx: &mut crate::CodegenCx<'tcx>,
    module: &mut dyn Module,
    instance: Instance<'tcx>,
) {
    let tcx = cx.tcx;

    let _inst_guard =
        crate::PrintOnPanic(|| format!("{:?} {}", instance, tcx.symbol_name(instance).name));
    debug_assert!(!instance.substs.needs_infer());

    let mir = tcx.instance_mir(instance.def);
    let _mir_guard = crate::PrintOnPanic(|| {
        let mut buf = Vec::new();
        rustc_middle::mir::write_mir_pretty(tcx, Some(instance.def_id()), &mut buf).unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    });

    // Declare function
    let symbol_name = tcx.symbol_name(instance);
    let sig = get_function_sig(tcx, module.isa().triple(), instance);
    let func_id = module.declare_function(symbol_name.name, Linkage::Local, &sig).unwrap();

    cx.cached_context.clear();

    // Make the FunctionBuilder
    let mut func_ctx = FunctionBuilderContext::new();
    let mut func = std::mem::replace(&mut cx.cached_context.func, Function::new());
    func.name = ExternalName::user(0, func_id.as_u32());
    func.signature = sig;
    func.collect_debug_info();

    let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Predefine blocks
    let start_block = bcx.create_block();
    let block_map: IndexVec<BasicBlock, Block> =
        (0..mir.basic_blocks().len()).map(|_| bcx.create_block()).collect();

    // Make FunctionCx
    let pointer_type = module.target_config().pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance);

    let mut fx = FunctionCx {
        cx,
        module,
        tcx,
        pointer_type,
        constants_cx: ConstantCx::new(),

        instance,
        symbol_name,
        mir,
        fn_abi: Some(FnAbi::of_instance(&RevealAllLayoutCx(tcx), instance, &[])),

        bcx,
        block_map,
        local_map: IndexVec::with_capacity(mir.local_decls.len()),
        caller_location: None, // set by `codegen_fn_prelude`

        clif_comments,
        source_info_set: indexmap::IndexSet::new(),
        next_ssa_var: 0,

        inline_asm_index: 0,
    };

    let arg_uninhabited = fx
        .mir
        .args_iter()
        .any(|arg| fx.layout_of(fx.monomorphize(&fx.mir.local_decls[arg].ty)).abi.is_uninhabited());

    if !crate::constant::check_constants(&mut fx) {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        crate::trap::trap_unreachable(&mut fx, "compilation should have been aborted");
    } else if arg_uninhabited {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        crate::trap::trap_unreachable(&mut fx, "function has uninhabited argument");
    } else {
        tcx.sess.time("codegen clif ir", || {
            tcx.sess
                .time("codegen prelude", || crate::abi::codegen_fn_prelude(&mut fx, start_block));
            codegen_fn_content(&mut fx);
        });
    }

    // Recover all necessary data from fx, before accessing func will prevent future access to it.
    let instance = fx.instance;
    let mut clif_comments = fx.clif_comments;
    let source_info_set = fx.source_info_set;
    let local_map = fx.local_map;

    fx.constants_cx.finalize(fx.tcx, &mut *fx.module);

    // Store function in context
    let context = &mut cx.cached_context;
    context.func = func;

    crate::pretty_clif::write_clif_file(
        tcx,
        "unopt",
        module.isa(),
        instance,
        &context,
        &clif_comments,
    );

    // Verify function
    verify_func(tcx, &clif_comments, &context.func);

    // If the return block is not reachable, then the SSA builder may have inserted an `iconst.i128`
    // instruction, which doesn't have an encoding.
    context.compute_cfg();
    context.compute_domtree();
    context.eliminate_unreachable_code(module.isa()).unwrap();
    context.dce(module.isa()).unwrap();
    // Some Cranelift optimizations expect the domtree to not yet be computed and as such don't
    // invalidate it when it would change.
    context.domtree.clear();

    // Perform rust specific optimizations
    tcx.sess.time("optimize clif ir", || {
        crate::optimize::optimize_function(
            tcx,
            module.isa(),
            instance,
            context,
            &mut clif_comments,
        );
    });

    // Define function
    tcx.sess.time("define function", || {
        context.want_disasm = crate::pretty_clif::should_write_ir(tcx);
        module
            .define_function(func_id, context, &mut NullTrapSink {}, &mut NullStackMapSink {})
            .unwrap()
    });

    // Write optimized function to file for debugging
    crate::pretty_clif::write_clif_file(
        tcx,
        "opt",
        module.isa(),
        instance,
        &context,
        &clif_comments,
    );

    if let Some(disasm) = &context.mach_compile_result.as_ref().unwrap().disasm {
        crate::pretty_clif::write_ir_file(
            tcx,
            || format!("{}.vcode", tcx.symbol_name(instance).name),
            |file| file.write_all(disasm.as_bytes()),
        )
    }

    // Define debuginfo for function
    let isa = module.isa();
    let debug_context = &mut cx.debug_context;
    let unwind_context = &mut cx.unwind_context;
    tcx.sess.time("generate debug info", || {
        if let Some(debug_context) = debug_context {
            debug_context.define_function(
                instance,
                func_id,
                symbol_name.name,
                isa,
                context,
                &source_info_set,
                local_map,
            );
        }
        unwind_context.add_function(func_id, &context, isa);
    });

    // Clear context to make it usable for the next function
    context.clear();
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
                    None,
                    Some(Box::new(writer)),
                    err,
                );
                tcx.sess.fatal(&format!("cranelift verify error:\n{}", pretty_error));
            }
        }
    });
}

fn codegen_fn_content(fx: &mut FunctionCx<'_, '_, '_>) {
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
            bb_data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
            let inst = fx.bcx.func.layout.last_inst(block).unwrap();
            fx.add_comment(inst, terminator_head);
        }

        fx.set_debug_loc(bb_data.terminator().source_info);

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
                // FIXME Mark failure block as cold once Cranelift supports it

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
                        let location = fx
                            .get_caller_location(bb_data.terminator().source_info.span)
                            .load_scalar(fx);

                        codegen_panic_inner(
                            fx,
                            rustc_hir::LangItem::PanicBoundsCheck,
                            &[index, len, location],
                            bb_data.terminator().source_info.span,
                        );
                    }
                    _ => {
                        let msg_str = msg.description();
                        codegen_panic(fx, msg_str, bb_data.terminator().source_info.span);
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
                fn_span,
                cleanup: _,
                from_hir_call: _,
            } => {
                fx.tcx.sess.time("codegen call", || {
                    crate::abi::codegen_terminator_call(fx, *fn_span, func, args, *destination)
                });
            }
            TerminatorKind::InlineAsm {
                template,
                operands,
                options,
                destination,
                line_spans: _,
            } => {
                crate::inline_asm::codegen_inline_asm(
                    fx,
                    bb_data.terminator().source_info.span,
                    template,
                    operands,
                    *options,
                );

                match *destination {
                    Some(destination) => {
                        let destination_block = fx.get_block(destination);
                        fx.bcx.ins().jump(destination_block, &[]);
                    }
                    None => {
                        crate::trap::trap_unreachable(
                            fx,
                            "[corruption] Returned from noreturn inline asm",
                        );
                    }
                }
            }
            TerminatorKind::Resume | TerminatorKind::Abort => {
                trap_unreachable(fx, "[corruption] Unwinding bb reached.");
            }
            TerminatorKind::Unreachable => {
                trap_unreachable(fx, "[corruption] Hit unreachable code.");
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
                crate::abi::codegen_drop(fx, bb_data.terminator().source_info.span, drop_place);

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

    #[cfg(disabled)]
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
                Rvalue::Cast(CastKind::Misc, ref operand, to_ty) => {
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
                    } else if let ty::Adt(adt_def, _substs) = from_ty.kind() {
                        // enum -> discriminant value
                        assert!(adt_def.is_enum());
                        match to_ty.kind() {
                            ty::Uint(_) | ty::Int(_) => {}
                            _ => unreachable!("cast adt {} -> {}", from_ty, to_ty),
                        }
                        let to_clif_ty = fx.clif_type(to_ty).unwrap();

                        let discriminant = crate::discriminant::codegen_get_discriminant(
                            fx,
                            operand,
                            fx.layout_of(operand.layout().ty.discriminant_ty(fx.tcx)),
                        )
                        .load_scalar(fx);

                        let res = crate::cast::clif_intcast(
                            fx,
                            discriminant,
                            to_clif_ty,
                            to_ty.is_signed(),
                        );
                        lval.write_cvalue(fx, CValue::by_val(res, dest_layout));
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
                    let discr =
                        crate::discriminant::codegen_get_discriminant(fx, value, dest_layout);
                    lval.write_cvalue(fx, discr);
                }
                Rvalue::Repeat(ref operand, times) => {
                    let operand = codegen_operand(fx, operand);
                    let times = fx
                        .monomorphize(times)
                        .eval(fx.tcx, ParamEnv::reveal_all())
                        .val
                        .try_to_bits(fx.tcx.data_layout.pointer_size)
                        .unwrap();
                    if operand.layout().size.bytes() == 0 {
                        // Do nothing for ZST's
                    } else if fx.clif_type(operand.layout().ty) == Some(types::I8) {
                        let times = fx.bcx.ins().iconst(fx.pointer_type, times as i64);
                        // FIXME use emit_small_memset where possible
                        let addr = lval.to_ptr().get_addr(fx);
                        let val = operand.load_scalar(fx);
                        fx.bcx.call_memset(fx.module.target_config(), addr, val, times);
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
                Rvalue::NullaryOp(NullOp::Box, content_ty) => {
                    let usize_type = fx.clif_type(fx.tcx.types.usize).unwrap();
                    let content_ty = fx.monomorphize(content_ty);
                    let layout = fx.layout_of(content_ty);
                    let llsize = fx.bcx.ins().iconst(usize_type, layout.size.bytes() as i64);
                    let llalign = fx.bcx.ins().iconst(usize_type, layout.align.abi.bytes() as i64);
                    let box_layout = fx.layout_of(fx.tcx.mk_box(content_ty));

                    // Allocate space:
                    let def_id =
                        match fx.tcx.lang_items().require(rustc_hir::LangItem::ExchangeMalloc) {
                            Ok(id) => id,
                            Err(s) => {
                                fx.tcx
                                    .sess
                                    .fatal(&format!("allocation of `{}` {}", box_layout.ty, s));
                            }
                        };
                    let instance = ty::Instance::mono(fx.tcx, def_id).polymorphize(fx.tcx);
                    let func_ref = fx.get_function_ref(instance);
                    let call = fx.bcx.ins().call(func_ref, &[llsize, llalign]);
                    let ptr = fx.bcx.inst_results(call)[0];
                    lval.write_cvalue(fx, CValue::by_val(ptr, box_layout));
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
                        NullOp::Box => unreachable!(),
                    };
                    let val =
                        CValue::const_val(fx, fx.layout_of(fx.tcx.types.usize), val.into());
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
        | StatementKind::Nop
        | StatementKind::FakeRead(..)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..) => {}

        StatementKind::LlvmInlineAsm(asm) => {
            match asm.asm.asm.as_str().trim() {
                "" => {
                    // Black box
                }
                _ => fx.tcx.sess.span_fatal(
                    stmt.source_info.span,
                    "Legacy `llvm_asm!` inline assembly is not supported. \
                    Try using the new `asm!` instead.",
                ),
            }
        }
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
            fx.bcx.call_memcpy(fx.module.target_config(), dst, src, bytes);
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
                        let elem_layout = fx.layout_of(elem_ty);
                        let ptr = cplace.to_ptr();
                        cplace = CPlace::for_ptr(
                            ptr.offset_i64(fx, elem_layout.size.bytes() as i64 * (from as i64)),
                            fx.layout_of(fx.tcx.mk_array(elem_ty, to - from)),
                        );
                    }
                    ty::Slice(elem_ty) => {
                        assert!(from_end, "slice subslices should be `from_end`");
                        let elem_layout = fx.layout_of(elem_ty);
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

pub(crate) fn codegen_panic<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, msg_str: &str, span: Span) {
    let location = fx.get_caller_location(span).load_scalar(fx);

    let msg_ptr = fx.anonymous_str(msg_str);
    let msg_len = fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(msg_str.len()).unwrap());
    let args = [msg_ptr, msg_len, location];

    codegen_panic_inner(fx, rustc_hir::LangItem::Panic, &args, span);
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

    crate::trap::trap_unreachable(fx, "panic lang item returned");
}
