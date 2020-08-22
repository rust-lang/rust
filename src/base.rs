use rustc_middle::ty::adjustment::PointerCast;
use rustc_index::vec::IndexVec;

use crate::prelude::*;

pub(crate) fn trans_fn<'tcx, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'tcx, B>,
    instance: Instance<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.codegen_cx.tcx;

    let mir = tcx.instance_mir(instance.def);

    // Declare function
    let (name, sig) = get_function_name_and_sig(tcx, cx.codegen_cx.module.isa().triple(), instance, false);
    let func_id = cx.codegen_cx.module.declare_function(&name, linkage, &sig).unwrap();

    // Make FunctionBuilder
    let context = &mut cx.cached_context;
    context.clear();
    context.func.name = ExternalName::user(0, func_id.as_u32());
    context.func.signature = sig;
    context.func.collect_debug_info();
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut context.func, &mut func_ctx);

    // Predefine blocks
    let start_block = bcx.create_block();
    let block_map: IndexVec<BasicBlock, Block> = (0..mir.basic_blocks().len()).map(|_| bcx.create_block()).collect();

    // Make FunctionCx
    let pointer_type = cx.codegen_cx.module.target_config().pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance);

    let mut fx = FunctionCx {
        tcx,
        module: &mut cx.codegen_cx.module,
        global_asm: &mut cx.global_asm,
        pointer_type,

        instance,
        mir,

        bcx,
        block_map,
        local_map: FxHashMap::with_capacity_and_hasher(mir.local_decls.len(), Default::default()),
        caller_location: None, // set by `codegen_fn_prelude`
        cold_blocks: EntitySet::new(),

        clif_comments,
        constants_cx: &mut cx.codegen_cx.constants_cx,
        vtables: &mut cx.vtables,
        source_info_set: indexmap::IndexSet::new(),
        next_ssa_var: 0,

        inline_asm_index: 0,
    };

    let arg_uninhabited = fx.mir.args_iter().any(|arg| fx.layout_of(fx.monomorphize(&fx.mir.local_decls[arg].ty)).abi.is_uninhabited());

    if arg_uninhabited {
        fx.bcx.append_block_params_for_function_params(fx.block_map[START_BLOCK]);
        fx.bcx.switch_to_block(fx.block_map[START_BLOCK]);
        crate::trap::trap_unreachable(&mut fx, "function has uninhabited argument");
    } else {
        tcx.sess.time("codegen clif ir", || {
            tcx.sess.time("codegen prelude", || crate::abi::codegen_fn_prelude(&mut fx, start_block));
            codegen_fn_content(&mut fx);
        });
    }

    // Recover all necessary data from fx, before accessing func will prevent future access to it.
    let instance = fx.instance;
    let mut clif_comments = fx.clif_comments;
    let source_info_set = fx.source_info_set;
    let local_map = fx.local_map;
    let cold_blocks = fx.cold_blocks;

    crate::pretty_clif::write_clif_file(
        cx.codegen_cx.tcx,
        "unopt",
        None,
        instance,
        &context,
        &clif_comments,
    );

    // Verify function
    verify_func(tcx, &clif_comments, &context.func);

    // Perform rust specific optimizations
    tcx.sess.time("optimize clif ir", || {
        crate::optimize::optimize_function(tcx, instance, context, &cold_blocks, &mut clif_comments);
    });

    // If the return block is not reachable, then the SSA builder may have inserted a `iconst.i128`
    // instruction, which doesn't have an encoding.
    context.compute_cfg();
    context.compute_domtree();
    context.eliminate_unreachable_code(cx.codegen_cx.module.isa()).unwrap();

    // Define function
    let module = &mut cx.codegen_cx.module;
    tcx.sess.time(
        "define function",
        || module.define_function(
            func_id,
            context,
            &mut cranelift_codegen::binemit::NullTrapSink {},
        ).unwrap(),
    );

    // Write optimized function to file for debugging
    crate::pretty_clif::write_clif_file(
        cx.codegen_cx.tcx,
        "opt",
        Some(cx.codegen_cx.module.isa()),
        instance,
        &context,
        &clif_comments,
    );

    // Define debuginfo for function
    let isa = cx.codegen_cx.module.isa();
    let debug_context = &mut cx.debug_context;
    let unwind_context = &mut cx.unwind_context;
    tcx.sess.time("generate debug info", || {
        if let Some(debug_context) = debug_context {
            debug_context.define_function(instance, func_id, &name, isa, context, &source_info_set, local_map);
        }
        unwind_context.add_function(func_id, &context, isa);
    });

    // Clear context to make it usable for the next function
    context.clear();
}

pub(crate) fn verify_func(tcx: TyCtxt<'_>, writer: &crate::pretty_clif::CommentWriter, func: &Function) {
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

fn codegen_fn_content(fx: &mut FunctionCx<'_, '_, impl Backend>) {
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
            trans_stmt(fx, block, stmt);
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
                if !fx.codegen_cx.tcx.sess.overflow_checks() {
                    if let mir::AssertKind::OverflowNeg(_) = *msg {
                        let target = fx.get_block(*target);
                        fx.bcx.ins().jump(target, &[]);
                        continue;
                    }
                }
                let cond = trans_operand(fx, cond).load_scalar(fx);

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

                let location = fx.get_caller_location(bb_data.terminator().source_info.span).load_scalar(fx);

                let args;
                let lang_item = match msg {
                    AssertKind::BoundsCheck { ref len, ref index } => {
                        let len = trans_operand(fx, len).load_scalar(fx);
                        let index = trans_operand(fx, index).load_scalar(fx);
                        args = [index, len, location];
                        rustc_hir::lang_items::PanicBoundsCheckFnLangItem
                    }
                    _ => {
                        let msg_str = msg.description();
                        let msg_ptr = fx.anonymous_str("assert", msg_str);
                        let msg_len = fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(msg_str.len()).unwrap());
                        args = [msg_ptr, msg_len, location];
                        rustc_hir::lang_items::PanicFnLangItem
                    }
                };

                let def_id = fx.codegen_cx.tcx.lang_items().require(lang_item).unwrap_or_else(|s| {
                    fx.codegen_cx.tcx.sess.span_fatal(bb_data.terminator().source_info.span, &s)
                });

                let instance = Instance::mono(fx.codegen_cx.tcx, def_id).polymorphize(fx.codegen_cx.tcx);
                let symbol_name = fx.codegen_cx.tcx.symbol_name(instance).name;

                fx.lib_call(&*symbol_name, vec![fx.pointer_type, fx.pointer_type, fx.pointer_type], vec![], &args);

                crate::trap::trap_unreachable(fx, "panic lang item returned");
            }

            TerminatorKind::SwitchInt {
                discr,
                switch_ty: _,
                values,
                targets,
            } => {
                let discr = trans_operand(fx, discr).load_scalar(fx);
                let mut switch = ::cranelift_frontend::Switch::new();
                for (i, value) in values.iter().enumerate() {
                    let block = fx.get_block(targets[i]);
                    switch.set_entry(*value, block);
                }
                let otherwise_block = fx.get_block(targets[targets.len() - 1]);
                switch.emit(&mut fx.bcx, discr, otherwise_block);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                fn_span,
                cleanup: _,
                from_hir_call: _,
            } => {
                fx.codegen_cx.tcx.sess.time("codegen call", || crate::abi::codegen_terminator_call(
                    fx,
                    *fn_span,
                    block,
                    func,
                    args,
                    *destination,
                ));
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
                        crate::trap::trap_unreachable(fx, "[corruption] Returned from noreturn inline asm");
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
                bug!("shouldn't exist at trans {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop {
                place,
                target,
                unwind: _,
            } => {
                let drop_place = trans_place(fx, *place);
                crate::abi::codegen_drop(fx, bb_data.terminator().source_info.span, drop_place);

                let target_block = fx.get_block(*target);
                fx.bcx.ins().jump(target_block, &[]);
            }
        };
    }

    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
}

fn trans_stmt<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    #[allow(unused_variables)]
    cur_block: Block,
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
            let place = trans_place(fx, **place);
            crate::discriminant::codegen_set_discriminant(fx, place, *variant_index);
        }
        StatementKind::Assign(to_place_and_rval) => {
            let lval = trans_place(fx, to_place_and_rval.0);
            let dest_layout = lval.layout();
            match &to_place_and_rval.1 {
                Rvalue::Use(operand) => {
                    let val = trans_operand(fx, operand);
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                    let place = trans_place(fx, *place);
                    place.write_place_ref(fx, lval);
                }
                Rvalue::ThreadLocalRef(def_id) => {
                    let val = crate::constant::codegen_tls_ref(fx, *def_id, lval.layout());
                    lval.write_cvalue(fx, val);
                }
                Rvalue::BinaryOp(bin_op, lhs, rhs) => {
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = crate::num::codegen_binop(fx, *bin_op, lhs, rhs);
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = if !fx.codegen_cx.tcx.sess.overflow_checks() {
                        let val =
                            crate::num::trans_int_binop(fx, *bin_op, lhs, rhs).load_scalar(fx);
                        let is_overflow = fx.bcx.ins().iconst(types::I8, 0);
                        CValue::by_val_pair(val, is_overflow, lval.layout())
                    } else {
                        crate::num::trans_checked_int_binop(fx, *bin_op, lhs, rhs)
                    };

                    lval.write_cvalue(fx, res);
                }
                Rvalue::UnaryOp(un_op, operand) => {
                    let operand = trans_operand(fx, operand);
                    let layout = operand.layout();
                    let val = operand.load_scalar(fx);
                    let res = match un_op {
                        UnOp::Not => {
                            match layout.ty.kind {
                                ty::Bool => {
                                    let res = fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0);
                                    CValue::by_val(fx.bcx.ins().bint(types::I8, res), layout)
                                }
                                ty::Uint(_) | ty::Int(_) => {
                                    CValue::by_val(fx.bcx.ins().bnot(val), layout)
                                }
                                _ => unreachable!("un op Not for {:?}", layout.ty),
                            }
                        }
                        UnOp::Neg => match layout.ty.kind {
                            ty::Int(IntTy::I128) => {
                                // FIXME remove this case once ineg.i128 works
                                let zero = CValue::const_val(fx, layout, 0);
                                crate::num::trans_int_binop(fx, BinOp::Sub, zero, operand)
                            }
                            ty::Int(_) => {
                                CValue::by_val(fx.bcx.ins().ineg(val), layout)
                            }
                            ty::Float(_) => {
                                CValue::by_val(fx.bcx.ins().fneg(val), layout)
                            }
                            _ => unreachable!("un op Neg for {:?}", layout.ty),
                        },
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::ReifyFnPointer), operand, to_ty) => {
                    let from_ty = fx.monomorphize(&operand.ty(&fx.mir.local_decls, fx.codegen_cx.tcx));
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    match from_ty.kind {
                        ty::FnDef(def_id, substs) => {
                            let func_ref = fx.get_function_ref(
                                Instance::resolve_for_fn_ptr(fx.codegen_cx.tcx, ParamEnv::reveal_all(), def_id, substs)
                                    .unwrap()
                                    .polymorphize(fx.codegen_cx.tcx),
                            );
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, to_layout));
                        }
                        _ => bug!("Trying to ReifyFnPointer on non FnDef {:?}", from_ty),
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::UnsafeFnPointer), operand, to_ty)
                | Rvalue::Cast(CastKind::Pointer(PointerCast::MutToConstPointer), operand, to_ty)
                | Rvalue::Cast(CastKind::Pointer(PointerCast::ArrayToPointer), operand, to_ty) => {
                    let to_layout = fx.layout_of(fx.monomorphize(to_ty));
                    let operand = trans_operand(fx, operand);
                    lval.write_cvalue(fx, operand.cast_pointer_to(to_layout));
                }
                Rvalue::Cast(CastKind::Misc, operand, to_ty) => {
                    let operand = trans_operand(fx, operand);
                    let from_ty = operand.layout().ty;
                    let to_ty = fx.monomorphize(to_ty);

                    fn is_fat_ptr<'tcx>(
                        fx: &FunctionCx<'_, 'tcx, impl Backend>,
                        ty: Ty<'tcx>,
                    ) -> bool {
                        ty.builtin_deref(true)
                            .map(
                                |ty::TypeAndMut {
                                     ty: pointee_ty,
                                     mutbl: _,
                                 }| has_ptr_meta(fx.codegen_cx.tcx, pointee_ty),
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
                    } else if let ty::Adt(adt_def, _substs) = from_ty.kind {
                        // enum -> discriminant value
                        assert!(adt_def.is_enum());
                        match to_ty.kind {
                            ty::Uint(_) | ty::Int(_) => {}
                            _ => unreachable!("cast adt {} -> {}", from_ty, to_ty),
                        }

                        use rustc_target::abi::{TagEncoding, Int, Variants};

                        match &operand.layout().variants {
                            Variants::Single { index } => {
                                let discr = operand.layout().ty.discriminant_for_variant(fx.codegen_cx.tcx, *index).unwrap();
                                let discr = if discr.ty.is_signed() {
                                    rustc_middle::mir::interpret::sign_extend(discr.val, fx.layout_of(discr.ty).size)
                                } else {
                                    discr.val
                                };

                                let discr = CValue::const_val(fx, fx.layout_of(to_ty), discr);
                                lval.write_cvalue(fx, discr);
                            }
                            Variants::Multiple {
                                tag,
                                tag_field,
                                tag_encoding: TagEncoding::Direct,
                                variants: _,
                            } => {
                                let cast_to = fx.clif_type(dest_layout.ty).unwrap();

                                // Read the tag/niche-encoded discriminant from memory.
                                let encoded_discr = operand.value_field(fx, mir::Field::new(*tag_field));
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
                            Variants::Multiple { ..} => unreachable!(),
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
                Rvalue::Cast(CastKind::Pointer(PointerCast::ClosureFnPointer(_)), operand, _to_ty) => {
                    let operand = trans_operand(fx, operand);
                    match operand.layout().ty.kind {
                        ty::Closure(def_id, substs) => {
                            let instance = Instance::resolve_closure(
                                fx.codegen_cx.tcx,
                                def_id,
                                substs,
                                ty::ClosureKind::FnOnce,
                            ).polymorphize(fx.codegen_cx.tcx);
                            let func_ref = fx.get_function_ref(instance);
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, lval.layout()));
                        }
                        _ => bug!("{} cannot be cast to a fn ptr", operand.layout().ty),
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), operand, _to_ty) => {
                    let operand = trans_operand(fx, operand);
                    operand.unsize_value(fx, lval);
                }
                Rvalue::Discriminant(place) => {
                    let place = trans_place(fx, *place);
                    let value = place.to_cvalue(fx);
                    let discr =
                        crate::discriminant::codegen_get_discriminant(fx, value, dest_layout);
                    lval.write_cvalue(fx, discr);
                }
                Rvalue::Repeat(operand, times) => {
                    let operand = trans_operand(fx, operand);
                    let times = fx
                        .monomorphize(times)
                        .eval(fx.codegen_cx.tcx, ParamEnv::reveal_all())
                        .val
                        .try_to_bits(fx.codegen_cx.tcx.data_layout.pointer_size)
                        .unwrap();
                    for i in 0..times {
                        let index = fx.bcx.ins().iconst(fx.pointer_type, i as i64);
                        let to = lval.place_index(fx, index);
                        to.write_cvalue(fx, operand);
                    }
                }
                Rvalue::Len(place) => {
                    let place = trans_place(fx, *place);
                    let usize_layout = fx.layout_of(fx.codegen_cx.tcx.types.usize);
                    let len = codegen_array_len(fx, place);
                    lval.write_cvalue(fx, CValue::by_val(len, usize_layout));
                }
                Rvalue::NullaryOp(NullOp::Box, content_ty) => {
                    use rustc_hir::lang_items::ExchangeMallocFnLangItem;

                    let usize_type = fx.clif_type(fx.codegen_cx.tcx.types.usize).unwrap();
                    let content_ty = fx.monomorphize(content_ty);
                    let layout = fx.layout_of(content_ty);
                    let llsize = fx.bcx.ins().iconst(usize_type, layout.size.bytes() as i64);
                    let llalign = fx
                        .bcx
                        .ins()
                        .iconst(usize_type, layout.align.abi.bytes() as i64);
                    let box_layout = fx.layout_of(fx.codegen_cx.tcx.mk_box(content_ty));

                    // Allocate space:
                    let def_id = match fx.codegen_cx.tcx.lang_items().require(ExchangeMallocFnLangItem) {
                        Ok(id) => id,
                        Err(s) => {
                            fx.codegen_cx.tcx
                                .sess
                                .fatal(&format!("allocation of `{}` {}", box_layout.ty, s));
                        }
                    };
                    let instance = ty::Instance::mono(fx.codegen_cx.tcx, def_id).polymorphize(fx.codegen_cx.tcx);
                    let func_ref = fx.get_function_ref(instance);
                    let call = fx.bcx.ins().call(func_ref, &[llsize, llalign]);
                    let ptr = fx.bcx.inst_results(call)[0];
                    lval.write_cvalue(fx, CValue::by_val(ptr, box_layout));
                }
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                    assert!(lval
                        .layout()
                        .ty
                        .is_sized(fx.codegen_cx.tcx.at(stmt.source_info.span), ParamEnv::reveal_all()));
                    let ty_size = fx.layout_of(fx.monomorphize(ty)).size.bytes();
                    let val = CValue::const_val(fx, fx.layout_of(fx.codegen_cx.tcx.types.usize), ty_size.into());
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Aggregate(kind, operands) => match **kind {
                    AggregateKind::Array(_ty) => {
                        for (i, operand) in operands.into_iter().enumerate() {
                            let operand = trans_operand(fx, operand);
                            let index = fx.bcx.ins().iconst(fx.pointer_type, i as i64);
                            let to = lval.place_index(fx, index);
                            to.write_cvalue(fx, operand);
                        }
                    }
                    _ => unreachable!("shouldn't exist at trans {:?}", to_place_and_rval.1),
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
                asm: asm_code, // Name
                outputs: output_names, // Vec<LlvmInlineAsmOutput>
                inputs: input_names,   // Vec<Name>
                clobbers,      // Vec<Name>
                volatile,      // bool
                alignstack,    // bool
                dialect: _,
                asm_str_style: _,
            } = asm;
            match asm_code.as_str().trim() {
                "" => {
                    // Black box
                }
                "mov %rbx, %rsi\n                  cpuid\n                  xchg %rbx, %rsi" => {
                    assert_eq!(input_names, &[Symbol::intern("{eax}"), Symbol::intern("{ecx}")]);
                    assert_eq!(output_names.len(), 4);
                    for (i, c) in (&["={eax}", "={esi}", "={ecx}", "={edx}"]).iter().enumerate() {
                        assert_eq!(&output_names[i].constraint.as_str(), c);
                        assert!(!output_names[i].is_rw);
                        assert!(!output_names[i].is_indirect);
                    }

                    assert_eq!(clobbers, &[]);

                    assert!(!volatile);
                    assert!(!alignstack);

                    assert_eq!(inputs.len(), 2);
                    let leaf = trans_operand(fx, &inputs[0].1).load_scalar(fx); // %eax
                    let subleaf = trans_operand(fx, &inputs[1].1).load_scalar(fx); // %ecx

                    let (eax, ebx, ecx, edx) = crate::intrinsics::codegen_cpuid_call(fx, leaf, subleaf);

                    assert_eq!(outputs.len(), 4);
                    trans_place(fx, outputs[0]).write_cvalue(fx, CValue::by_val(eax, fx.layout_of(fx.codegen_cx.tcx.types.u32)));
                    trans_place(fx, outputs[1]).write_cvalue(fx, CValue::by_val(ebx, fx.layout_of(fx.codegen_cx.tcx.types.u32)));
                    trans_place(fx, outputs[2]).write_cvalue(fx, CValue::by_val(ecx, fx.layout_of(fx.codegen_cx.tcx.types.u32)));
                    trans_place(fx, outputs[3]).write_cvalue(fx, CValue::by_val(edx, fx.layout_of(fx.codegen_cx.tcx.types.u32)));
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
                _ if fx.codegen_cx.tcx.symbol_name(fx.instance).name.starts_with("___chkstk") => {
                    crate::trap::trap_unimplemented(fx, "Stack probes are not supported");
                }
                _ if fx.codegen_cx.tcx.symbol_name(fx.instance).name == "__alloca" => {
                    crate::trap::trap_unimplemented(fx, "Alloca is not supported");
                }
                // Used in sys::windows::abort_internal
                "int $$0x29" => {
                    crate::trap::trap_unimplemented(fx, "Windows abort");
                }
                _ => fx.codegen_cx.tcx.sess.span_fatal(stmt.source_info.span, "Inline assembly is not supported"),
            }
        }
    }
}

fn codegen_array_len<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
) -> Value {
    match place.layout().ty.kind {
        ty::Array(_elem_ty, len) => {
            let len = fx.monomorphize(&len)
                .eval(fx.codegen_cx.tcx, ParamEnv::reveal_all())
                .eval_usize(fx.codegen_cx.tcx, ParamEnv::reveal_all()) as i64;
            fx.bcx.ins().iconst(fx.pointer_type, len)
        }
        ty::Slice(_elem_ty) => place
            .to_ptr_maybe_unsized()
            .1
            .expect("Length metadata for slice place"),
        _ => bug!("Rvalue::Len({:?})", place),
    }
}

pub(crate) fn trans_place<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
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
                let index = if !from_end {
                    fx.bcx.ins().iconst(fx.pointer_type, i64::from(offset))
                } else {
                    let len = codegen_array_len(fx, cplace);
                    fx.bcx.ins().iadd_imm(len, -i64::from(offset))
                };
                cplace = cplace.place_index(fx, index);
            }
            PlaceElem::Subslice { from, to, from_end } => {
                // These indices are generated by slice patterns.
                // slice[from:-to] in Python terms.

                match cplace.layout().ty.kind {
                    ty::Array(elem_ty, _len) => {
                        assert!(!from_end, "array subslices are never `from_end`");
                        let elem_layout = fx.layout_of(elem_ty);
                        let ptr = cplace.to_ptr();
                        cplace = CPlace::for_ptr(
                            ptr.offset_i64(fx, elem_layout.size.bytes() as i64 * i64::from(from)),
                            fx.layout_of(fx.codegen_cx.tcx.mk_array(elem_ty, u64::from(to) - u64::from(from))),
                        );
                    }
                    ty::Slice(elem_ty) => {
                        assert!(from_end, "slice subslices should be `from_end`");
                        let elem_layout = fx.layout_of(elem_ty);
                        let (ptr, len) = cplace.to_ptr_maybe_unsized();
                        let len = len.unwrap();
                        cplace = CPlace::for_ptr_with_extra(
                            ptr.offset_i64(fx, elem_layout.size.bytes() as i64 * i64::from(from)),
                            fx.bcx.ins().iadd_imm(len, -(i64::from(from) + i64::from(to))),
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

pub(crate) fn trans_operand<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    operand: &Operand<'tcx>,
) -> CValue<'tcx> {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => {
            let cplace = trans_place(fx, *place);
            cplace.to_cvalue(fx)
        }
        Operand::Constant(const_) => crate::constant::trans_constant(fx, const_),
    }
}
