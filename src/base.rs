use rustc::ty::adjustment::PointerCast;

use crate::prelude::*;

pub fn trans_fn<'a, 'clif, 'tcx: 'a, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'clif, 'tcx, B>,
    instance: Instance<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;

    let mir = tcx.instance_mir(instance.def);

    // Declare function
    let (name, sig) = get_function_name_and_sig(tcx, instance, false);
    let func_id = cx.module.declare_function(&name, linkage, &sig).unwrap();
    let mut debug_context = cx
        .debug_context
        .as_mut()
        .map(|debug_context| FunctionDebugContext::new(tcx, debug_context, mir, &name, &sig));

    // Make FunctionBuilder
    let mut func = Function::with_name_signature(ExternalName::user(0, 0), sig);
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Predefine ebb's
    let start_ebb = bcx.create_ebb();
    let mut ebb_map: HashMap<BasicBlock, Ebb> = HashMap::new();
    for (bb, _bb_data) in mir.basic_blocks().iter_enumerated() {
        ebb_map.insert(bb, bcx.create_ebb());
    }

    // Make FunctionCx
    let pointer_type = cx.module.target_config().pointer_type();
    let clif_comments = crate::pretty_clif::CommentWriter::new(tcx, instance);

    let mut fx = FunctionCx {
        tcx,
        module: cx.module,
        pointer_type,

        instance,
        mir,

        bcx,
        ebb_map,
        local_map: HashMap::new(),

        clif_comments,
        constants: &mut cx.ccx,
        caches: &mut cx.caches,
        source_info_set: indexmap::IndexSet::new(),
    };

    with_unimpl_span(fx.mir.span, || {
        crate::abi::codegen_fn_prelude(&mut fx, start_ebb);
        codegen_fn_content(&mut fx);
    });

    // Recover all necessary data from fx, before accessing func will prevent future access to it.
    let instance = fx.instance;
    let clif_comments = fx.clif_comments;
    let source_info_set = fx.source_info_set;

    #[cfg(debug_assertions)]
    crate::pretty_clif::write_clif_file(cx.tcx, "unopt", instance, &func, &clif_comments, None);

    // Verify function
    verify_func(tcx, &clif_comments, &func);

    // Define function
    let context = &mut cx.caches.context;
    context.func = func;
    cx.module
        .define_function(func_id, context)
        .unwrap();

    let value_ranges = context.build_value_labels_ranges(cx.module.isa()).expect("value location ranges");

    // Write optimized function to file for debugging
    #[cfg(debug_assertions)]
    crate::pretty_clif::write_clif_file(cx.tcx, "opt", instance, &context.func, &clif_comments, Some(&value_ranges));

    // Define debuginfo for function
    let isa = cx.module.isa();
    debug_context
        .as_mut()
        .map(|x| x.define(tcx, context, isa, &source_info_set));

    // Clear context to make it usable for the next function
    context.clear();
}

fn verify_func(tcx: TyCtxt, writer: &crate::pretty_clif::CommentWriter, func: &Function) {
    let flags = settings::Flags::new(settings::builder());
    match ::cranelift::codegen::verify_function(&func, &flags) {
        Ok(_) => {}
        Err(err) => {
            tcx.sess.err(&format!("{:?}", err));
            let pretty_error = ::cranelift::codegen::print_errors::pretty_verifier_error(
                &func,
                None,
                Some(Box::new(writer)),
                err,
            );
            tcx.sess
                .fatal(&format!("cranelift verify error:\n{}", pretty_error));
        }
    }
}

fn codegen_fn_content<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx, impl Backend>) {
    for (bb, bb_data) in fx.mir.basic_blocks().iter_enumerated() {
        if bb_data.is_cleanup {
            // Unwinding after panicking is not supported
            continue;
        }

        let ebb = fx.get_ebb(bb);
        fx.bcx.switch_to_block(ebb);

        fx.bcx.ins().nop();
        for stmt in &bb_data.statements {
            fx.set_debug_loc(stmt.source_info);
            trans_stmt(fx, ebb, stmt);
        }

        #[cfg(debug_assertions)]
        {
            let mut terminator_head = "\n".to_string();
            bb_data
                .terminator()
                .kind
                .fmt_head(&mut terminator_head)
                .unwrap();
            let inst = fx.bcx.func.layout.last_inst(ebb).unwrap();
            fx.add_comment(inst, terminator_head);
        }

        fx.set_debug_loc(bb_data.terminator().source_info);

        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                let ebb = fx.get_ebb(*target);
                fx.bcx.ins().jump(ebb, &[]);
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
                let cond = trans_operand(fx, cond).load_scalar(fx);
                // TODO HACK brz/brnz for i8/i16 is not yet implemented
                let cond = fx.bcx.ins().uextend(types::I32, cond);
                let target = fx.get_ebb(*target);
                if *expected {
                    fx.bcx.ins().brnz(cond, target, &[]);
                } else {
                    fx.bcx.ins().brz(cond, target, &[]);
                };
                trap_panic(fx, format!("[panic] Assert {:?} at {:?} failed.", msg, bb_data.terminator().source_info.span));
            }

            TerminatorKind::SwitchInt {
                discr,
                switch_ty: _,
                values,
                targets,
            } => {
                let discr = trans_operand(fx, discr).load_scalar(fx);
                let mut switch = ::cranelift::frontend::Switch::new();
                for (i, value) in values.iter().enumerate() {
                    let ebb = fx.get_ebb(targets[i]);
                    switch.set_entry(*value as u64, ebb);
                }
                let otherwise_ebb = fx.get_ebb(targets[targets.len() - 1]);
                switch.emit(&mut fx.bcx, discr, otherwise_ebb);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                cleanup: _,
                from_hir_call: _,
            } => {
                crate::abi::codegen_terminator_call(fx, func, args, destination);
            }
            TerminatorKind::Resume | TerminatorKind::Abort => {
                trap_unreachable(fx, "[corruption] Unwinding bb reached.");
            }
            TerminatorKind::Unreachable => {
                trap_unreachable(fx, "[corruption] Hit unreachable code.");
            }
            TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::GeneratorDrop => {
                bug!("shouldn't exist at trans {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop {
                location,
                target,
                unwind: _,
            } => {
                let drop_place = trans_place(fx, location);
                crate::abi::codegen_drop(fx, drop_place);

                let target_ebb = fx.get_ebb(*target);
                fx.bcx.ins().jump(target_ebb, &[]);
            }
        };
    }

    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
}

fn trans_stmt<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    cur_ebb: Ebb,
    stmt: &Statement<'tcx>,
) {
    let _print_guard = PrintOnPanic(|| format!("stmt {:?}", stmt));

    fx.set_debug_loc(stmt.source_info);

    #[cfg(debug_assertions)]
    match &stmt.kind {
        StatementKind::StorageLive(..) | StatementKind::StorageDead(..) => {} // Those are not very useful
        _ => {
            let inst = fx.bcx.func.layout.last_inst(cur_ebb).unwrap();
            fx.add_comment(inst, format!("{:?}", stmt));
        }
    }

    match &stmt.kind {
        StatementKind::SetDiscriminant {
            place,
            variant_index,
        } => {
            let place = trans_place(fx, place);
            let layout = place.layout();
            if layout.for_variant(&*fx, *variant_index).abi == layout::Abi::Uninhabited {
                return;
            }
            match layout.variants {
                layout::Variants::Single { index } => {
                    assert_eq!(index, *variant_index);
                }
                layout::Variants::Multiple {
                    discr: _,
                    discr_index,
                    discr_kind: layout::DiscriminantKind::Tag,
                    variants: _,
                } => {
                    let ptr = place.place_field(fx, mir::Field::new(discr_index));
                    let to = layout
                        .ty
                        .discriminant_for_variant(fx.tcx, *variant_index)
                        .unwrap()
                        .val;
                    let discr = CValue::const_val(fx, ptr.layout().ty, to);
                    ptr.write_cvalue(fx, discr);
                }
                layout::Variants::Multiple {
                    discr: _,
                    discr_index,
                    discr_kind: layout::DiscriminantKind::Niche {
                        dataful_variant,
                        ref niche_variants,
                        niche_start,
                    },
                    variants: _,
                } => {
                    if *variant_index != dataful_variant {
                        let niche = place.place_field(fx, mir::Field::new(discr_index));
                        //let niche_llty = niche.layout.immediate_llvm_type(bx.cx);
                        let niche_value =
                            ((variant_index.as_u32() - niche_variants.start().as_u32()) as u128)
                                .wrapping_add(niche_start);
                        // FIXME(eddyb) Check the actual primitive type here.
                        let niche_llval = if niche_value == 0 {
                            CValue::const_val(fx, niche.layout().ty, 0)
                        } else {
                            CValue::const_val(fx, niche.layout().ty, niche_value)
                        };
                        niche.write_cvalue(fx, niche_llval);
                    }
                }
            }
        }
        StatementKind::Assign(to_place, rval) => {
            let lval = trans_place(fx, to_place);
            let dest_layout = lval.layout();
            match &**rval {
                Rvalue::Use(operand) => {
                    let val = trans_operand(fx, operand);
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Ref(_, _, place) => {
                    let place = trans_place(fx, place);
                    place.write_place_ref(fx, lval);
                }
                Rvalue::BinaryOp(bin_op, lhs, rhs) => {
                    let ty = fx.monomorphize(&lhs.ty(fx.mir, fx.tcx));
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = match ty.sty {
                        ty::Bool => trans_bool_binop(fx, *bin_op, lhs, rhs),
                        ty::Uint(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
                        }
                        ty::Int(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true)
                        }
                        ty::Float(_) => trans_float_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::Char => trans_char_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::RawPtr(..) => trans_ptr_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::FnPtr(..) => trans_ptr_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        _ => unimplemented!("binop {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
                    let ty = fx.monomorphize(&lhs.ty(fx.mir, fx.tcx));
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = match ty.sty {
                        ty::Uint(_) => {
                            trans_checked_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
                        }
                        ty::Int(_) => {
                            trans_checked_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true)
                        }
                        _ => unimplemented!("checked binop {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::UnaryOp(un_op, operand) => {
                    let operand = trans_operand(fx, operand);
                    let layout = operand.layout();
                    let val = operand.load_scalar(fx);
                    let res = match un_op {
                        UnOp::Not => {
                            match layout.ty.sty {
                                ty::Bool => {
                                    let val = fx.bcx.ins().uextend(types::I32, val); // WORKAROUND for CraneStation/cranelift#466
                                    let res = fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0);
                                    fx.bcx.ins().bint(types::I8, res)
                                }
                                ty::Uint(_) | ty::Int(_) => {
                                    fx.bcx.ins().bnot(val)
                                }
                                _ => unimplemented!("un op Not for {:?}", layout.ty),
                            }
                        }
                        UnOp::Neg => match layout.ty.sty {
                            ty::Int(_) => {
                                let clif_ty = fx.clif_type(layout.ty).unwrap();
                                if clif_ty == types::I128 {
                                    crate::trap::trap_unreachable_ret_value(fx, layout, "i128 neg is not yet supported").load_scalar(fx)
                                } else {
                                    let zero = fx.bcx.ins().iconst(clif_ty, 0);
                                    fx.bcx.ins().isub(zero, val)
                                }
                            }
                            ty::Float(_) => fx.bcx.ins().fneg(val),
                            _ => unimplemented!("un op Neg for {:?}", layout.ty),
                        },
                    };
                    lval.write_cvalue(fx, CValue::by_val(res, layout));
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::ReifyFnPointer), operand, ty) => {
                    let layout = fx.layout_of(ty);
                    match fx
                        .monomorphize(&operand.ty(&fx.mir.local_decls, fx.tcx))
                        .sty
                    {
                        ty::FnDef(def_id, substs) => {
                            let func_ref = fx.get_function_ref(
                                Instance::resolve(fx.tcx, ParamEnv::reveal_all(), def_id, substs)
                                    .unwrap(),
                            );
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, layout));
                        }
                        _ => bug!("Trying to ReifyFnPointer on non FnDef {:?}", ty),
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::UnsafeFnPointer), operand, ty)
                | Rvalue::Cast(CastKind::Pointer(PointerCast::MutToConstPointer), operand, ty) => {
                    let operand = trans_operand(fx, operand);
                    let layout = fx.layout_of(ty);
                    lval.write_cvalue(fx, operand.unchecked_cast_to(layout));
                }
                Rvalue::Cast(CastKind::Misc, operand, to_ty) => {
                    let operand = trans_operand(fx, operand);
                    let from_ty = operand.layout().ty;

                    fn is_fat_ptr<'a, 'tcx: 'a>(fx: &FunctionCx<'a, 'tcx, impl Backend>, ty: Ty<'tcx>) -> bool {
                        ty
                            .builtin_deref(true)
                            .map(|ty::TypeAndMut {ty: pointee_ty, mutbl: _ }| fx.layout_of(pointee_ty).is_unsized())
                            .unwrap_or(false)
                    }

                    if is_fat_ptr(fx, from_ty) {
                        if is_fat_ptr(fx, to_ty) {
                            // fat-ptr -> fat-ptr
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        } else {
                            // fat-ptr -> thin-ptr
                            let (ptr, _extra) = operand.load_scalar_pair(fx);
                            lval.write_cvalue(fx, CValue::by_val(ptr, dest_layout))
                        }
                    } else if let ty::Adt(adt_def, _substs) = from_ty.sty {
                        // enum -> discriminant value
                        assert!(adt_def.is_enum());
                        match to_ty.sty {
                            ty::Uint(_) | ty::Int(_) => {},
                            _ => unreachable!("cast adt {} -> {}", from_ty, to_ty),
                        }

                        // FIXME avoid forcing to stack
                        let place =
                            CPlace::for_addr(operand.force_stack(fx), operand.layout());
                        let discr = trans_get_discriminant(fx, place, fx.layout_of(to_ty));
                        lval.write_cvalue(fx, discr);
                    } else {
                        let from_clif_ty = fx.clif_type(from_ty).unwrap();
                        let to_clif_ty = fx.clif_type(to_ty).unwrap();
                        let from = operand.load_scalar(fx);

                        let signed = match from_ty.sty {
                            ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::Char | ty::Uint(..) | ty::Bool => false,
                            ty::Int(..) => true,
                            ty::Float(..) => false, // `signed` is unused for floats
                            _ => panic!("{}", from_ty),
                        };

                        let res = if from_clif_ty.is_int() && to_clif_ty.is_int() {
                            // int-like -> int-like
                            crate::common::clif_intcast(
                                fx,
                                from,
                                to_clif_ty,
                                signed,
                            )
                        } else if from_clif_ty.is_int() && to_clif_ty.is_float() {
                            // int-like -> float
                            if signed {
                                fx.bcx.ins().fcvt_from_sint(to_clif_ty, from)
                            } else {
                                fx.bcx.ins().fcvt_from_uint(to_clif_ty, from)
                            }
                        } else if from_clif_ty.is_float() && to_clif_ty.is_int() {
                            // float -> int-like
                            let from = operand.load_scalar(fx);
                            if signed {
                                fx.bcx.ins().fcvt_to_sint_sat(to_clif_ty, from)
                            } else {
                                fx.bcx.ins().fcvt_to_uint_sat(to_clif_ty, from)
                            }
                        } else if from_clif_ty.is_float() && to_clif_ty.is_float() {
                            // float -> float
                            match (from_clif_ty, to_clif_ty) {
                                (types::F32, types::F64) => {
                                    fx.bcx.ins().fpromote(types::F64, from)
                                }
                                (types::F64, types::F32) => {
                                    fx.bcx.ins().fdemote(types::F32, from)
                                }
                                _ => from,
                            }
                        } else {
                            unimpl!("rval misc {:?} {:?}", from_ty, to_ty)
                        };
                        lval.write_cvalue(fx, CValue::by_val(res, dest_layout));
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::ClosureFnPointer(_)), operand, _ty) => {
                    let operand = trans_operand(fx, operand);
                    match operand.layout().ty.sty {
                        ty::Closure(def_id, substs) => {
                            let instance = Instance::resolve_closure(
                                fx.tcx,
                                def_id,
                                substs,
                                ty::ClosureKind::FnOnce,
                            );
                            let func_ref = fx.get_function_ref(instance);
                            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
                            lval.write_cvalue(fx, CValue::by_val(func_addr, lval.layout()));
                        }
                        _ => {
                            bug!("{} cannot be cast to a fn ptr", operand.layout().ty)
                        }
                    }
                }
                Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), operand, _ty) => {
                    let operand = trans_operand(fx, operand);
                    operand.unsize_value(fx, lval);
                }
                Rvalue::Discriminant(place) => {
                    let place = trans_place(fx, place);
                    let discr = trans_get_discriminant(fx, place, dest_layout);
                    lval.write_cvalue(fx, discr);
                }
                Rvalue::Repeat(operand, times) => {
                    let operand = trans_operand(fx, operand);
                    for i in 0..*times {
                        let index = fx.bcx.ins().iconst(fx.pointer_type, i as i64);
                        let to = lval.place_index(fx, index);
                        to.write_cvalue(fx, operand);
                    }
                }
                Rvalue::Len(place) => {
                    let place = trans_place(fx, place);
                    let usize_layout = fx.layout_of(fx.tcx.types.usize);
                    let len = codegen_array_len(fx, place);
                    lval.write_cvalue(fx, CValue::by_val(len, usize_layout));
                }
                Rvalue::NullaryOp(NullOp::Box, content_ty) => {
                    use rustc::middle::lang_items::ExchangeMallocFnLangItem;

                    let usize_type = fx.clif_type(fx.tcx.types.usize).unwrap();
                    let layout = fx.layout_of(content_ty);
                    let llsize = fx.bcx.ins().iconst(usize_type, layout.size.bytes() as i64);
                    let llalign = fx
                        .bcx
                        .ins()
                        .iconst(usize_type, layout.align.abi.bytes() as i64);
                    let box_layout = fx.layout_of(fx.tcx.mk_box(content_ty));

                    // Allocate space:
                    let def_id = match fx.tcx.lang_items().require(ExchangeMallocFnLangItem) {
                        Ok(id) => id,
                        Err(s) => {
                            fx.tcx
                                .sess
                                .fatal(&format!("allocation of `{}` {}", box_layout.ty, s));
                        }
                    };
                    let instance = ty::Instance::mono(fx.tcx, def_id);
                    let func_ref = fx.get_function_ref(instance);
                    let call = fx.bcx.ins().call(func_ref, &[llsize, llalign]);
                    let ptr = fx.bcx.inst_results(call)[0];
                    lval.write_cvalue(fx, CValue::by_val(ptr, box_layout));
                }
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                    assert!(lval
                        .layout()
                        .ty
                        .is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all()));
                    let ty_size = fx.layout_of(ty).size.bytes();
                    let val = CValue::const_val(fx, fx.tcx.types.usize, ty_size.into());
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
                    _ => unimpl!("shouldn't exist at trans {:?}", rval),
                },
            }
        }
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Nop
        | StatementKind::FakeRead(..)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..) => {}

        StatementKind::InlineAsm(asm) => {
            use syntax::ast::Name;
            let InlineAsm { asm, outputs: _, inputs: _ } = &**asm;
            let rustc::hir::InlineAsm {
                asm: asm_code, // Name
                outputs, // Vec<Name>
                inputs, // Vec<Name>
                clobbers, // Vec<Name>
                volatile, // bool
                alignstack, // bool
                dialect: _, // syntax::ast::AsmDialect
                asm_str_style: _,
                ctxt: _,
            } = asm;
            match &*asm_code.as_str() {
                "cpuid" | "cpuid\n" => {
                    assert_eq!(inputs, &[Name::intern("{eax}"), Name::intern("{ecx}")]);

                    assert_eq!(outputs.len(), 4);
                    for (i, c) in (&["={eax}", "={ebx}", "={ecx}", "={edx}"]).iter().enumerate() {
                        assert_eq!(&outputs[i].constraint.as_str(), c);
                        assert!(!outputs[i].is_rw);
                        assert!(!outputs[i].is_indirect);
                    }

                    assert_eq!(clobbers, &[Name::intern("rbx")]);

                    assert!(!volatile);
                    assert!(!alignstack);

                    crate::trap::trap_unimplemented(fx, "__cpuid_count arch intrinsic is not supported");
                }
                "xgetbv" => {
                    assert_eq!(inputs, &[Name::intern("{ecx}")]);

                    assert_eq!(outputs.len(), 2);
                    for (i, c) in (&["={eax}", "={edx}"]).iter().enumerate() {
                        assert_eq!(&outputs[i].constraint.as_str(), c);
                        assert!(!outputs[i].is_rw);
                        assert!(!outputs[i].is_indirect);
                    }

                    assert_eq!(clobbers, &[]);

                    assert!(!volatile);
                    assert!(!alignstack);

                    crate::trap::trap_unimplemented(fx, "_xgetbv arch intrinsic is not supported");
                }
                _ if fx.tcx.symbol_name(fx.instance).as_str() == "__rust_probestack" => {
                    crate::trap::trap_unimplemented(fx, "__rust_probestack is not supported");
                }
                _ => unimpl!("Inline assembly is not supported"),
            }
        }
    }
}

fn codegen_array_len<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
) -> Value {
    match place.layout().ty.sty {
        ty::Array(_elem_ty, len) => {
            let len = crate::constant::force_eval_const(fx, len).unwrap_usize(fx.tcx) as i64;
            fx.bcx.ins().iconst(fx.pointer_type, len)
        }
        ty::Slice(_elem_ty) => place
            .to_addr_maybe_unsized(fx)
            .1
            .expect("Length metadata for slice place"),
        _ => bug!("Rvalue::Len({:?})", place),
    }
}

pub fn trans_get_discriminant<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
    dest_layout: TyLayout<'tcx>,
) -> CValue<'tcx> {
    let layout = place.layout();

    if layout.abi == layout::Abi::Uninhabited {
        return trap_unreachable_ret_value(fx, dest_layout, "[panic] Tried to get discriminant for uninhabited type.");
    }

    let (discr_scalar, discr_index, discr_kind) = match &layout.variants {
        layout::Variants::Single { index } => {
            let discr_val = layout
                .ty
                .ty_adt_def()
                .map_or(index.as_u32() as u128, |def| {
                    def.discriminant_for_variant(fx.tcx, *index).val
                });
            return CValue::const_val(fx, dest_layout.ty, discr_val);
        }
        layout::Variants::Multiple { discr, discr_index, discr_kind, variants: _ } => {
            (discr, *discr_index, discr_kind)
        }
    };

    let discr = place.place_field(fx, mir::Field::new(discr_index)).to_cvalue(fx);
    let discr_ty = discr.layout().ty;
    let lldiscr = discr.load_scalar(fx);
    match discr_kind {
        layout::DiscriminantKind::Tag => {
            let signed = match discr_scalar.value {
                layout::Int(_, signed) => signed,
                _ => false,
            };
            let val = clif_intcast(fx, lldiscr, fx.clif_type(dest_layout.ty).unwrap(), signed);
            return CValue::by_val(val, dest_layout);
        }
        layout::DiscriminantKind::Niche {
            dataful_variant,
            ref niche_variants,
            niche_start,
        } => {
            let niche_llty = fx.clif_type(discr_ty).unwrap();
            let dest_clif_ty = fx.clif_type(dest_layout.ty).unwrap();
            if niche_variants.start() == niche_variants.end() {
                let b = fx
                    .bcx
                    .ins()
                    .icmp_imm(IntCC::Equal, lldiscr, *niche_start as u64 as i64);
                let if_true = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, niche_variants.start().as_u32() as i64);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, dataful_variant.as_u32() as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::by_val(val, dest_layout);
            } else {
                // Rebase from niche values to discriminant values.
                let delta = niche_start.wrapping_sub(niche_variants.start().as_u32() as u128);
                let delta = fx.bcx.ins().iconst(niche_llty, delta as u64 as i64);
                let lldiscr = fx.bcx.ins().isub(lldiscr, delta);
                let b = fx.bcx.ins().icmp_imm(
                    IntCC::UnsignedLessThanOrEqual,
                    lldiscr,
                    niche_variants.end().as_u32() as i64,
                );
                let if_true =
                    clif_intcast(fx, lldiscr, fx.clif_type(dest_layout.ty).unwrap(), false);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, dataful_variant.as_u32() as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::by_val(val, dest_layout);
            }
        }
    }
}

macro_rules! binop_match {
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, bug) => {
        bug!("binop {} on {} lhs: {:?} rhs: {:?}", stringify!($var), $bug_fmt, $lhs, $rhs)
    };
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, icmp($cc:ident)) => {{
        assert_eq!($fx.tcx.types.bool, $ret_ty);
        let ret_layout = $fx.layout_of($ret_ty);

        let b = $fx.bcx.ins().icmp(IntCC::$cc, $lhs, $rhs);
        CValue::by_val($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, fcmp($cc:ident)) => {{
        assert_eq!($fx.tcx.types.bool, $ret_ty);
        let ret_layout = $fx.layout_of($ret_ty);
        let b = $fx.bcx.ins().fcmp(FloatCC::$cc, $lhs, $rhs);
        CValue::by_val($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, custom(|| $body:expr)) => {{
        $body
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $name:ident) => {{
        let ret_layout = $fx.layout_of($ret_ty);
        CValue::by_val($fx.bcx.ins().$name($lhs, $rhs), ret_layout)
    }};
    (
        $fx:expr, $bin_op:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $bug_fmt:expr;
        $(
            $var:ident ($sign:pat) $name:tt $( ( $($next:tt)* ) )? ;
        )*
    ) => {{
        let lhs = $lhs.load_scalar($fx);
        let rhs = $rhs.load_scalar($fx);
        match ($bin_op, $signed) {
            $(
                (BinOp::$var, $sign) => binop_match!(@single $fx, $bug_fmt, $var, $signed, lhs, rhs, $ret_ty, $name $( ( $($next)* ) )?),
            )*
        }
    }}
}

fn trans_bool_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, fx.tcx.types.bool, "bool";
        Add (_) bug;
        Sub (_) bug;
        Mul (_) bug;
        Div (_) bug;
        Rem (_) bug;
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) icmp(Equal);
        Lt (_) icmp(UnsignedLessThan);
        Le (_) icmp(UnsignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (_) icmp(UnsignedGreaterThanOrEqual);
        Gt (_) icmp(UnsignedGreaterThan);

        Offset (_) bug;
    };

    res
}

pub fn trans_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    out_ty: Ty<'tcx>,
    signed: bool,
) -> CValue<'tcx> {
    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            lhs.layout().ty,
            rhs.layout().ty,
            "int binop requires lhs and rhs of same type"
        );
    }

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, false, signed, lhs, rhs, out_ty) {
        return res;
    }

    binop_match! {
        fx, bin_op, signed, lhs, rhs, out_ty, "int/uint";
        Add (_) iadd;
        Sub (_) isub;
        Mul (_) imul;
        Div (false) udiv;
        Div (true) sdiv;
        Rem (false) urem;
        Rem (true) srem;
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) ishl;
        Shr (false) ushr;
        Shr (true) sshr;

        Eq (_) icmp(Equal);
        Lt (false) icmp(UnsignedLessThan);
        Lt (true) icmp(SignedLessThan);
        Le (false) icmp(UnsignedLessThanOrEqual);
        Le (true) icmp(SignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (false) icmp(UnsignedGreaterThanOrEqual);
        Ge (true) icmp(SignedGreaterThanOrEqual);
        Gt (false) icmp(UnsignedGreaterThan);
        Gt (true) icmp(SignedGreaterThan);

        Offset (_) bug;
    }
}

pub fn trans_checked_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
    out_ty: Ty<'tcx>,
    signed: bool,
) -> CValue<'tcx> {
    if !fx.tcx.sess.overflow_checks() {
        return trans_int_binop(fx, bin_op, in_lhs, in_rhs, out_ty, signed);
    }

    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            in_lhs.layout().ty,
            in_rhs.layout().ty,
            "checked int binop requires lhs and rhs of same type"
        );
    }

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, true, signed, in_lhs, in_rhs, out_ty) {
        return res;
    }

    let (res, has_overflow) = match bin_op {
        BinOp::Add => {
            /*let (val, c_out) = fx.bcx.ins().iadd_cout(lhs, rhs);
            (val, c_out)*/
            // FIXME(CraneStation/cranelift#849) legalize iadd_cout for i8 and i16
            let val = fx.bcx.ins().iadd(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedLessThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let slt = fx.bcx.ins().icmp(IntCC::SignedLessThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, slt)
            };
            (val, has_overflow)
        }
        BinOp::Sub => {
            /*let (val, b_out) = fx.bcx.ins().isub_bout(lhs, rhs);
            (val, b_out)*/
            // FIXME(CraneStation/cranelift#849) legalize isub_bout for i8 and i16
            let val = fx.bcx.ins().isub(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let sgt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, sgt)
            };
            (val, has_overflow)
        }
        BinOp::Mul => {
            let val = fx.bcx.ins().imul(lhs, rhs);
            /*let val_hi = if !signed {
                fx.bcx.ins().umulhi(lhs, rhs)
            } else {
                fx.bcx.ins().smulhi(lhs, rhs)
            };
            let has_overflow = fx.bcx.ins().icmp_imm(IntCC::NotEqual, val_hi, 0);*/
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        BinOp::Shl => {
            let val = fx.bcx.ins().ishl(lhs, rhs);
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        BinOp::Shr => {
            let val = if !signed {
                fx.bcx.ins().ushr(lhs, rhs)
            } else {
                fx.bcx.ins().sshr(lhs, rhs)
            };
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        _ => bug!(
            "binop {:?} on checked int/uint lhs: {:?} rhs: {:?}",
            bin_op,
            in_lhs,
            in_rhs
        ),
    };

    let has_overflow = fx.bcx.ins().bint(types::I8, has_overflow);
    let out_place = CPlace::new_stack_slot(fx, out_ty);
    let out_layout = out_place.layout();
    out_place.write_cvalue(fx, CValue::by_val_pair(res, has_overflow, out_layout));

    out_place.to_cvalue(fx)
}

fn trans_float_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "float";
        Add (_) fadd;
        Sub (_) fsub;
        Mul (_) fmul;
        Div (_) fdiv;
        Rem (_) custom(|| {
            assert_eq!(lhs.layout().ty, ty);
            assert_eq!(rhs.layout().ty, ty);
            match ty.sty {
                ty::Float(FloatTy::F32) => fx.easy_call("fmodf", &[lhs, rhs], ty),
                ty::Float(FloatTy::F64) => fx.easy_call("fmod", &[lhs, rhs], ty),
                _ => bug!(),
            }
        });
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) fcmp(Equal);
        Lt (_) fcmp(LessThan);
        Le (_) fcmp(LessThanOrEqual);
        Ne (_) fcmp(NotEqual);
        Ge (_) fcmp(GreaterThanOrEqual);
        Gt (_) fcmp(GreaterThan);

        Offset (_) bug;
    };

    res
}

fn trans_char_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "char";
        Add (_) bug;
        Sub (_) bug;
        Mul (_) bug;
        Div (_) bug;
        Rem (_) bug;
        BitXor (_) bug;
        BitAnd (_) bug;
        BitOr (_) bug;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) icmp(Equal);
        Lt (_) icmp(UnsignedLessThan);
        Le (_) icmp(UnsignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (_) icmp(UnsignedGreaterThanOrEqual);
        Gt (_) icmp(UnsignedGreaterThan);

        Offset (_) bug;
    };

    res
}

fn trans_ptr_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ret_ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let not_fat = match lhs.layout().ty.sty {
        ty::RawPtr(TypeAndMut { ty, mutbl: _ }) => {
            ty.is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all())
        }
        ty::FnPtr(..) => true,
        _ => bug!("trans_ptr_binop on non ptr"),
    };
    if not_fat {
        if let BinOp::Offset = bin_op {
            let (base, offset) = (lhs, rhs.load_scalar(fx));
            let pointee_ty = base.layout().ty.builtin_deref(true).unwrap().ty;
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
            let base_val = base.load_scalar(fx);
            let res = fx.bcx.ins().iadd(base_val, ptr_diff);
            return CValue::by_val(res, base.layout());
        }

        binop_match! {
            fx, bin_op, false, lhs, rhs, ret_ty, "ptr";
            Add (_) bug;
            Sub (_) bug;
            Mul (_) bug;
            Div (_) bug;
            Rem (_) bug;
            BitXor (_) bug;
            BitAnd (_) bug;
            BitOr (_) bug;
            Shl (_) bug;
            Shr (_) bug;

            Eq (_) icmp(Equal);
            Lt (_) icmp(UnsignedLessThan);
            Le (_) icmp(UnsignedLessThanOrEqual);
            Ne (_) icmp(NotEqual);
            Ge (_) icmp(UnsignedGreaterThanOrEqual);
            Gt (_) icmp(UnsignedGreaterThan);

            Offset (_) bug; // Handled above
        }
    } else {
        let (lhs_ptr, lhs_extra) = lhs.load_scalar_pair(fx);
        let (rhs_ptr, rhs_extra) = rhs.load_scalar_pair(fx);

        let res = match bin_op {
            BinOp::Eq => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);
                let extra_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_extra, rhs_extra);
                fx.bcx.ins().band(ptr_eq, extra_eq)
            }
            BinOp::Ne => {
                let ptr_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_ptr, rhs_ptr);
                let extra_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_extra, rhs_extra);
                fx.bcx.ins().bor(ptr_ne, extra_ne)
            }
            BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);

                let ptr_cmp = fx.bcx.ins().icmp(match bin_op {
                    BinOp::Lt => IntCC::UnsignedLessThan,
                    BinOp::Le => IntCC::UnsignedLessThanOrEqual,
                    BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                    BinOp::Gt => IntCC::UnsignedGreaterThan,
                    _ => unreachable!(),
                }, lhs_ptr, rhs_ptr);

                let extra_cmp = fx.bcx.ins().icmp(match bin_op {
                    BinOp::Lt => IntCC::UnsignedLessThan,
                    BinOp::Le => IntCC::UnsignedLessThanOrEqual,
                    BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                    BinOp::Gt => IntCC::UnsignedGreaterThan,
                    _ => unreachable!(),
                }, lhs_extra, rhs_extra);

                fx.bcx.ins().select(ptr_eq, extra_cmp, ptr_cmp)
            }
            _ => panic!("bin_op {:?} on ptr", bin_op),
        };

        assert_eq!(fx.tcx.types.bool, ret_ty);
        let ret_layout = fx.layout_of(ret_ty);
        CValue::by_val(fx.bcx.ins().bint(types::I8, res), ret_layout)
    }
}

pub fn trans_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    place: &Place<'tcx>,
) -> CPlace<'tcx> {
    let base = match &place.base {
        PlaceBase::Local(local) => fx.get_local_place(*local),
        PlaceBase::Static(static_) => match static_.kind {
            StaticKind::Static(def_id) => {
                crate::constant::codegen_static_ref(fx, def_id, static_.ty)
            }
            StaticKind::Promoted(promoted) => {
                crate::constant::trans_promoted(fx, promoted, static_.ty)
            }
        }
    };

    trans_place_projection(fx, base, &place.projection)
}

pub fn trans_place_projection<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    base: CPlace<'tcx>,
    projection: &Option<Box<Projection<'tcx>>>,
) -> CPlace<'tcx> {
    let projection = if let Some(projection) = projection {
        projection
    } else {
        return base;
    };

    let base = trans_place_projection(fx, base, &projection.base);

    match projection.elem {
        ProjectionElem::Deref => base.place_deref(fx),
        ProjectionElem::Field(field, _ty) => base.place_field(fx, field),
        ProjectionElem::Index(local) => {
            let index = fx.get_local_place(local).to_cvalue(fx).load_scalar(fx);
            base.place_index(fx, index)
        }
        ProjectionElem::ConstantIndex {
            offset,
            min_length: _,
            from_end,
        } => {
            let index = if !from_end {
                fx.bcx.ins().iconst(fx.pointer_type, offset as i64)
            } else {
                let len = codegen_array_len(fx, base);
                fx.bcx.ins().iadd_imm(len, -(offset as i64))
            };
            base.place_index(fx, index)
        }
        ProjectionElem::Subslice { from, to } => {
            // These indices are generated by slice patterns.
            // slice[from:-to] in Python terms.

            match base.layout().ty.sty {
                ty::Array(elem_ty, len) => {
                    let elem_layout = fx.layout_of(elem_ty);
                    let ptr = base.to_addr(fx);
                    let len = crate::constant::force_eval_const(fx, len).unwrap_usize(fx.tcx);
                    CPlace::for_addr(
                        fx.bcx.ins().iadd_imm(ptr, elem_layout.size.bytes() as i64 * from as i64),
                        fx.layout_of(fx.tcx.mk_array(elem_ty, len - from as u64 - to as u64)),
                    )
                }
                ty::Slice(elem_ty) => {
                    let elem_layout = fx.layout_of(elem_ty);
                    let (ptr, len) = base.to_addr_maybe_unsized(fx);
                    let len = len.unwrap();
                    CPlace::for_addr_with_extra(
                        fx.bcx.ins().iadd_imm(ptr, elem_layout.size.bytes() as i64 * from as i64),
                        fx.bcx.ins().iadd_imm(len, -(from as i64 + to as i64)),
                        base.layout(),
                    )
                }
                _ => unreachable!(),
            }
        }
        ProjectionElem::Downcast(_adt_def, variant) => base.downcast_variant(fx, variant),
    }
}

pub fn trans_operand<'a, 'tcx>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    operand: &Operand<'tcx>,
) -> CValue<'tcx> {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => {
            let cplace = trans_place(fx, place);
            cplace.to_cvalue(fx)
        }
        Operand::Constant(const_) => crate::constant::trans_constant(fx, const_),
    }
}
