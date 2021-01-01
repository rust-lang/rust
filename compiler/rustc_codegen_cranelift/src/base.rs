//! Codegen of a single function

use rustc_index::vec::IndexVec;
use rustc_middle::ty::adjustment::PointerCast;

use crate::prelude::*;

pub(crate) fn codegen_fn<'tcx>(
    cx: &mut crate::CodegenCx<'tcx, impl Module>,
    instance: Instance<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;

    let _inst_guard =
        crate::PrintOnPanic(|| format!("{:?} {}", instance, tcx.symbol_name(instance).name));
    debug_assert!(!instance.substs.needs_infer());

    let mir = tcx.instance_mir(instance.def);

    // Declare function
    let (name, sig) = get_function_name_and_sig(tcx, cx.module.isa().triple(), instance, false);
    let func_id = cx.module.declare_function(&name, linkage, &sig).unwrap();

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
    let block_map: IndexVec<BasicBlock, Block> = (0..mir.basic_blocks().len())
        .map(|_| bcx.create_block())
        .collect();

    // Make FunctionCx
    let pointer_type = cx.module.target_config().pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance);

    let mut fx = FunctionCx {
        cx,
        tcx,
        pointer_type,

        instance,
        mir,

        bcx,
        block_map,
        local_map: IndexVec::with_capacity(mir.local_decls.len()),
        caller_location: None, // set by `codegen_fn_prelude`
        cold_blocks: EntitySet::new(),

        clif_comments,
        source_info_set: indexmap::IndexSet::new(),
        next_ssa_var: 0,

        inline_asm_index: 0,
    };

    let arg_uninhabited = fx.mir.args_iter().any(|arg| {
        fx.layout_of(fx.monomorphize(&fx.mir.local_decls[arg].ty))
            .abi
            .is_uninhabited()
    });

    if arg_uninhabited {
        fx.bcx
            .append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        crate::trap::trap_unreachable(&mut fx, "function has uninhabited argument");
    } else {
        tcx.sess.time("codegen clif ir", || {
            tcx.sess.time("codegen prelude", || {
                crate::abi::codegen_fn_prelude(&mut fx, start_block)
            });
            codegen_fn_content(&mut fx);
        });
    }

    // Recover all necessary data from fx, before accessing func will prevent future access to it.
    let instance = fx.instance;
    let mut clif_comments = fx.clif_comments;
    let source_info_set = fx.source_info_set;
    let local_map = fx.local_map;
    let cold_blocks = fx.cold_blocks;

    // Store function in context
    let context = &mut cx.cached_context;
    context.func = func;

    crate::pretty_clif::write_clif_file(tcx, "unopt", None, instance, &context, &clif_comments);

    // Verify function
    verify_func(tcx, &clif_comments, &context.func);

    // Perform rust specific optimizations
    tcx.sess.time("optimize clif ir", || {
        crate::optimize::optimize_function(
            tcx,
            instance,
            context,
            &cold_blocks,
            &mut clif_comments,
        );
    });

    // If the return block is not reachable, then the SSA builder may have inserted an `iconst.i128`
    // instruction, which doesn't have an encoding.
    context.compute_cfg();
    context.compute_domtree();
    context.eliminate_unreachable_code(cx.module.isa()).unwrap();
    context.dce(cx.module.isa()).unwrap();

    context.want_disasm = crate::pretty_clif::should_write_ir(tcx);

    // Define function
    let module = &mut cx.module;
    tcx.sess.time("define function", || {
        module
            .define_function(
                func_id,
                context,
                &mut cranelift_codegen::binemit::NullTrapSink {},
            )
            .unwrap()
    });

    // Write optimized function to file for debugging
    crate::pretty_clif::write_clif_file(
        tcx,
        "opt",
        Some(cx.module.isa()),
        instance,
        &context,
        &clif_comments,
    );

    if let Some(mach_compile_result) = &context.mach_compile_result {
        if let Some(disasm) = &mach_compile_result.disasm {
            crate::pretty_clif::write_ir_file(
                tcx,
                &format!("{}.vcode", tcx.symbol_name(instance).name),
                |file| file.write_all(disasm.as_bytes()),
            )
        }
    }

    // Define debuginfo for function
    let isa = cx.module.isa();
    let debug_context = &mut cx.debug_context;
    let unwind_context = &mut cx.unwind_context;
    tcx.sess.time("generate debug info", || {
        if let Some(debug_context) = debug_context {
            debug_context.define_function(
                instance,
                func_id,
                &name,
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
                tcx.sess
                    .fatal(&format!("cranelift verify error:\n{}", pretty_error));
            }
        }
    });
}

fn codegen_fn_content(fx: &mut FunctionCx<'_, '_, impl Module>) {
    crate::constant::check_constants(fx);

    for (bb, bb_data) in fx.mir.basic_blocks().iter_enumerated() {
        let block = fx.get_block(bb);
        fx.bcx.switch_to_block(block);

        if bb_data.is_cleanup {
            // Unwinding after panicking is not supported
            continue;

            // FIXME once unwinding is supported uncomment next lines
            // // Unwinding is unlikely to happen, so mark cleanup block's as cold.
            // fx.cold_blocks.insert(block);
        }

        fx.bcx.ins().nop();
        for stmt in &bb_data.statements {
            fx.set_debug_loc(stmt.source_info);
            codegen_stmt(fx, block, stmt);
        }

        #[cfg(debug_assertions)]
        {
            let mut terminator_head = "\n".to_string();
            bb_data
                .terminator()
                .kind
                .fmt_head(&mut terminator_head)
                .unwrap();
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
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                cleanup: _,
            } => {
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
                fx.cold_blocks.insert(failure);

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

            TerminatorKind::SwitchInt {
                discr,
                switch_ty,
                targets,
            } => {
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
                    let discr =
                        crate::optimize::peephole::make_branchable_value(&mut fx.bcx, discr);
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
                    crate::abi::codegen_terminator_call(
                        fx,
                        *fn_span,
                        block,
                        func,
                        args,
                        *destination,
                    )
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
            TerminatorKind::Drop {
                place,
                target,
                unwind: _,
            } => {
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
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    #[allow(unused_variables)] cur_block: Block,
    stmt: &Statement<'tcx>,
) {
    let _print_guard = crate::PrintOnPanic(|| format!("stmt {:?}", stmt));

    fx.set_debug_loc(stmt.source_info);

    #[cfg(false_debug_assertions)]
    match &stmt.kind {
        StatementKind::StorageLive(..) | StatementKind::StorageDead(..) => {} // Those are not very useful
        _ => {
            let inst = fx.bcx.func.layout.last_inst(cur_block).unwrap();
            fx.add_comment(inst, format!("{:?}", stmt));
        }
    }

    match &stmt.kind {
        StatementKind::SetDiscriminant {
            place,
            variant_index,
        } => {
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
                Rvalue::BinaryOp(bin_op, ref lhs, ref rhs) => {
                    let lhs = codegen_operand(fx, lhs);
                    let rhs = codegen_operand(fx, rhs);

                    let res = crate::num::codegen_binop(fx, bin_op, lhs, rhs);
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, ref lhs, ref rhs) => {
                    let lhs = codegen_operand(fx, lhs);
                    let rhs = codegen_operand(fx, rhs);

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

                    fn is_fat_ptr<'tcx>(
                        fx: &FunctionCx<'_, 'tcx, impl Module>,
                        ty: Ty<'tcx>,
                    ) -> bool {
                        ty.builtin_deref(true)
                            .map(
                                |ty::TypeAndMut {
                                     ty: pointee_ty,
                                     mutbl: _,
                                 }| {
                                    has_ptr_meta(fx.tcx, pointee_ty)
                                },
                            )
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

                        use rustc_target::abi::{Int, TagEncoding, Variants};

                        match operand.layout().variants {
                            Variants::Single { index } => {
                                let discr = operand
                                    .layout()
                                    .ty
                                    .discriminant_for_variant(fx.tcx, index)
                                    .unwrap();
                                let discr = if discr.ty.is_signed() {
                                    fx.layout_of(discr.ty).size.sign_extend(discr.val)
                                } else {
                                    discr.val
                                };
                                let discr = discr.into();

                                let discr = CValue::const_val(fx, fx.layout_of(to_ty), discr);
                                lval.write_cvalue(fx, discr);
                            }
                            Variants::Multiple {
                                ref tag,
                                tag_field,
                                tag_encoding: TagEncoding::Direct,
                                variants: _,
                            } => {
                                let cast_to = fx.clif_type(dest_layout.ty).unwrap();

                                // Read the tag/niche-encoded discriminant from memory.
                                let encoded_discr =
                                    operand.value_field(fx, mir::Field::new(tag_field));
                                let encoded_discr = encoded_discr.load_scalar(fx);

                                // Decode the discriminant (specifically if it's niche-encoded).
                                let signed = match tag.value {
                                    Int(_, signed) => signed,
                                    _ => false,
                                };
                                let val = clif_intcast(fx, encoded_discr, cast_to, signed);
                                let val = CValue::by_val(val, dest_layout);
                                lval.write_cvalue(fx, val);
                            }
                            Variants::Multiple { .. } => unreachable!(),
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
                    if fx.clif_type(operand.layout().ty) == Some(types::I8) {
                        let times = fx.bcx.ins().iconst(fx.pointer_type, times as i64);
                        // FIXME use emit_small_memset where possible
                        let addr = lval.to_ptr().get_addr(fx);
                        let val = operand.load_scalar(fx);
                        fx.bcx
                            .call_memset(fx.cx.module.target_config(), addr, val, times);
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
                    let llalign = fx
                        .bcx
                        .ins()
                        .iconst(usize_type, layout.align.abi.bytes() as i64);
                    let box_layout = fx.layout_of(fx.tcx.mk_box(content_ty));

                    // Allocate space:
                    let def_id = match fx
                        .tcx
                        .lang_items()
                        .require(rustc_hir::LangItem::ExchangeMalloc)
                    {
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
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                    assert!(lval
                        .layout()
                        .ty
                        .is_sized(fx.tcx.at(stmt.source_info.span), ParamEnv::reveal_all()));
                    let ty_size = fx.layout_of(fx.monomorphize(ty)).size.bytes();
                    let val =
                        CValue::const_val(fx, fx.layout_of(fx.tcx.types.usize), ty_size.into());
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
            use rustc_span::symbol::Symbol;
            let LlvmInlineAsm {
                asm,
                outputs,
                inputs,
            } = &**asm;
            let rustc_hir::LlvmInlineAsmInner {
                asm: asm_code,         // Name
                outputs: output_names, // Vec<LlvmInlineAsmOutput>
                inputs: input_names,   // Vec<Name>
                clobbers,              // Vec<Name>
                volatile,              // bool
                alignstack,            // bool
                dialect: _,
                asm_str_style: _,
            } = asm;
            match asm_code.as_str().trim() {
                "" => {
                    // Black box
                }
                "mov %rbx, %rsi\n                  cpuid\n                  xchg %rbx, %rsi" => {
                    assert_eq!(
                        input_names,
                        &[Symbol::intern("{eax}"), Symbol::intern("{ecx}")]
                    );
                    assert_eq!(output_names.len(), 4);
                    for (i, c) in (&["={eax}", "={esi}", "={ecx}", "={edx}"])
                        .iter()
                        .enumerate()
                    {
                        assert_eq!(&output_names[i].constraint.as_str(), c);
                        assert!(!output_names[i].is_rw);
                        assert!(!output_names[i].is_indirect);
                    }

                    assert_eq!(clobbers, &[]);

                    assert!(!volatile);
                    assert!(!alignstack);

                    assert_eq!(inputs.len(), 2);
                    let leaf = codegen_operand(fx, &inputs[0].1).load_scalar(fx); // %eax
                    let subleaf = codegen_operand(fx, &inputs[1].1).load_scalar(fx); // %ecx

                    let (eax, ebx, ecx, edx) =
                        crate::intrinsics::codegen_cpuid_call(fx, leaf, subleaf);

                    assert_eq!(outputs.len(), 4);
                    codegen_place(fx, outputs[0])
                        .write_cvalue(fx, CValue::by_val(eax, fx.layout_of(fx.tcx.types.u32)));
                    codegen_place(fx, outputs[1])
                        .write_cvalue(fx, CValue::by_val(ebx, fx.layout_of(fx.tcx.types.u32)));
                    codegen_place(fx, outputs[2])
                        .write_cvalue(fx, CValue::by_val(ecx, fx.layout_of(fx.tcx.types.u32)));
                    codegen_place(fx, outputs[3])
                        .write_cvalue(fx, CValue::by_val(edx, fx.layout_of(fx.tcx.types.u32)));
                }
                "xgetbv" => {
                    assert_eq!(input_names, &[Symbol::intern("{ecx}")]);

                    assert_eq!(output_names.len(), 2);
                    for (i, c) in (&["={eax}", "={edx}"]).iter().enumerate() {
                        assert_eq!(&output_names[i].constraint.as_str(), c);
                        assert!(!output_names[i].is_rw);
                        assert!(!output_names[i].is_indirect);
                    }

                    assert_eq!(clobbers, &[]);

                    assert!(!volatile);
                    assert!(!alignstack);

                    crate::trap::trap_unimplemented(fx, "_xgetbv arch intrinsic is not supported");
                }
                // ___chkstk, ___chkstk_ms and __alloca are only used on Windows
                _ if fx
                    .tcx
                    .symbol_name(fx.instance)
                    .name
                    .starts_with("___chkstk") =>
                {
                    crate::trap::trap_unimplemented(fx, "Stack probes are not supported");
                }
                _ if fx.tcx.symbol_name(fx.instance).name == "__alloca" => {
                    crate::trap::trap_unimplemented(fx, "Alloca is not supported");
                }
                // Used in sys::windows::abort_internal
                "int $$0x29" => {
                    crate::trap::trap_unimplemented(fx, "Windows abort");
                }
                _ => fx
                    .tcx
                    .sess
                    .span_fatal(stmt.source_info.span, "Inline assembly is not supported"),
            }
        }
        StatementKind::Coverage { .. } => fx.tcx.sess.fatal("-Zcoverage is unimplemented"),
    }
}

fn codegen_array_len<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    place: CPlace<'tcx>,
) -> Value {
    match *place.layout().ty.kind() {
        ty::Array(_elem_ty, len) => {
            let len = fx
                .monomorphize(len)
                .eval_usize(fx.tcx, ParamEnv::reveal_all()) as i64;
            fx.bcx.ins().iconst(fx.pointer_type, len)
        }
        ty::Slice(_elem_ty) => place
            .to_ptr_maybe_unsized()
            .1
            .expect("Length metadata for slice place"),
        _ => bug!("Rvalue::Len({:?})", place),
    }
}

pub(crate) fn codegen_place<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
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
            PlaceElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } => {
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
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
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
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    msg_str: &str,
    span: Span,
) {
    let location = fx.get_caller_location(span).load_scalar(fx);

    let msg_ptr = fx.anonymous_str("assert", msg_str);
    let msg_len = fx
        .bcx
        .ins()
        .iconst(fx.pointer_type, i64::try_from(msg_str.len()).unwrap());
    let args = [msg_ptr, msg_len, location];

    codegen_panic_inner(fx, rustc_hir::LangItem::Panic, &args, span);
}

pub(crate) fn codegen_panic_inner<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    lang_item: rustc_hir::LangItem,
    args: &[Value],
    span: Span,
) {
    let def_id = fx
        .tcx
        .lang_items()
        .require(lang_item)
        .unwrap_or_else(|s| fx.tcx.sess.span_fatal(span, &s));

    let instance = Instance::mono(fx.tcx, def_id).polymorphize(fx.tcx);
    let symbol_name = fx.tcx.symbol_name(instance).name;

    fx.lib_call(
        &*symbol_name,
        vec![fx.pointer_type, fx.pointer_type, fx.pointer_type],
        vec![],
        args,
    );

    crate::trap::trap_unreachable(fx, "panic lang item returned");
}
