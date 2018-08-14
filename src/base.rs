use crate::prelude::*;

pub fn trans_mono_item<'a, 'tcx: 'a>(
    cx: &mut CodegenCx<'a, 'tcx, CurrentBackend>,
    context: &mut Context,
    mono_item: MonoItem<'tcx>,
) {
    let tcx = cx.tcx;

    match mono_item {
        MonoItem::Fn(inst) => match inst {
            Instance {
                def: InstanceDef::Item(def_id),
                substs: _,
            } => {
                let mut mir = ::std::io::Cursor::new(Vec::new());
                ::rustc_mir::util::write_mir_pretty(tcx, Some(def_id), &mut mir).unwrap();
                tcx.sess.warn(&format!(
                    "{:?}:\n\n{}",
                    inst,
                    String::from_utf8_lossy(&mir.into_inner())
                ));

                let (func_id, mut func) = cx.predefine_function(inst);

                let comments = trans_fn(cx, &mut func, inst);

                let mut writer = crate::pretty_clif::CommentWriter(comments);
                let mut cton = String::new();
                ::cranelift::codegen::write::decorate_function(&mut writer, &mut cton, &func, None)
                    .unwrap();
                tcx.sess.warn(&cton);

                let flags = settings::Flags::new(settings::builder());
                match ::cranelift::codegen::verify_function(&func, &flags) {
                    Ok(_) => {}
                    Err(err) => {
                        tcx.sess.err(&format!("{:?}", err));
                        let pretty_error =
                            ::cranelift::codegen::print_errors::pretty_verifier_error(
                                &func,
                                None,
                                Some(Box::new(writer)),
                                &err,
                            );
                        tcx.sess
                            .fatal(&format!("cretonne verify error:\n{}", pretty_error));
                    }
                }

                context.func = func;
                // TODO: cranelift doesn't yet support some of the things needed
                if should_codegen(cx.tcx) {
                    cx.module.define_function(func_id, context).unwrap();
                    cx.defined_functions.push(func_id);
                }

                context.clear();
            }
            Instance {
                def: InstanceDef::DropGlue(_, _),
                substs: _,
            } => unimpl!("Unimplemented drop glue instance"),
            inst => unimpl!("Unimplemented instance {:?}", inst),
        },
        MonoItem::Static(def_id) => {
            crate::constant::codegen_static(cx, def_id);
        }
        MonoItem::GlobalAsm(node_id) => cx
            .tcx
            .sess
            .fatal(&format!("Unimplemented global asm mono item {:?}", node_id)),
    }
}

pub fn trans_fn<'a, 'tcx: 'a>(
    cx: &mut CodegenCx<'a, 'tcx, CurrentBackend>,
    f: &mut Function,
    instance: Instance<'tcx>,
) -> HashMap<Inst, String> {
    let mir = cx.tcx.optimized_mir(instance.def_id());
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx: FunctionBuilder<Variable> = FunctionBuilder::new(f, &mut func_ctx);

    let start_ebb = bcx.create_ebb();
    bcx.switch_to_block(start_ebb);
    let mut ebb_map: HashMap<BasicBlock, Ebb> = HashMap::new();
    for (bb, _bb_data) in mir.basic_blocks().iter_enumerated() {
        ebb_map.insert(bb, bcx.create_ebb());
    }

    let mut fx = FunctionCx {
        tcx: cx.tcx,
        module: &mut cx.module,
        instance,
        mir,
        bcx,
        param_substs: {
            assert!(!instance.substs.needs_infer());
            instance.substs
        },
        ebb_map,
        local_map: HashMap::new(),
        comments: HashMap::new(),
        constants: &mut cx.constants,
    };
    let fx = &mut fx;

    crate::abi::codegen_fn_prelude(fx, start_ebb);

    fx.bcx
        .ins()
        .jump(*fx.ebb_map.get(&START_BLOCK).unwrap(), &[]);

    for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
        let ebb = fx.get_ebb(bb);
        fx.bcx.switch_to_block(ebb);

        fx.bcx.ins().nop();
        for stmt in &bb_data.statements {
            trans_stmt(fx, ebb, stmt);
        }

        let mut terminator_head = "\n".to_string();
        bb_data
            .terminator()
            .kind
            .fmt_head(&mut terminator_head)
            .unwrap();
        let inst = fx.bcx.func.layout.last_inst(ebb).unwrap();
        fx.add_comment(inst, terminator_head);

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
                msg: _,
                target,
                cleanup: _,
            } => {
                let cond = trans_operand(fx, cond).load_value(fx);
                // TODO HACK brz/brnz for i8/i16 is not yet implemented
                let cond = fx.bcx.ins().uextend(types::I32, cond);
                let target = fx.get_ebb(*target);
                if *expected {
                    fx.bcx.ins().brnz(cond, target, &[]);
                } else {
                    fx.bcx.ins().brz(cond, target, &[]);
                };
                fx.bcx.ins().trap(TrapCode::User(!0));
            }

            TerminatorKind::SwitchInt {
                discr,
                switch_ty: _,
                values,
                targets,
            } => {
                fx.bcx.ins().trap(TrapCode::User(0));
                // TODO: prevent panics on large and negative disciminants
                if should_codegen(fx.tcx) {
                    let discr = trans_operand(fx, discr).load_value(fx);
                    let mut jt_data = JumpTableData::new();
                    for (i, value) in values.iter().enumerate() {
                        let ebb = fx.get_ebb(targets[i]);
                        jt_data.set_entry(*value as usize, ebb);
                    }
                    let jump_table = fx.bcx.create_jump_table(jt_data);
                    fx.bcx.ins().br_table(discr, jump_table);
                    let otherwise_ebb = fx.get_ebb(targets[targets.len() - 1]);
                    fx.bcx.ins().jump(otherwise_ebb, &[]);
                }
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                cleanup: _,
            } => {
                crate::abi::codegen_call(fx, func, args, destination);
            }
            TerminatorKind::Resume | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                fx.bcx.ins().trap(TrapCode::User(!0));
            }
            TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                bug!("shouldn't exist at trans {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop { target, .. } | TerminatorKind::DropAndReplace { target, .. } => {
                // TODO call drop impl
                // unimplemented!("terminator {:?}", bb_data.terminator());
                let target_ebb = fx.get_ebb(*target);
                fx.bcx.ins().jump(target_ebb, &[]);
            }
            TerminatorKind::GeneratorDrop => {
                unimplemented!("terminator GeneratorDrop");
            }
        };
    }

    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();

    fx.comments.clone()
}

fn trans_stmt<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, cur_ebb: Ebb, stmt: &Statement<'tcx>) {
    fx.tcx.sess.warn(&format!("stmt {:?}", stmt));

    let inst = fx.bcx.func.layout.last_inst(cur_ebb).unwrap();
    fx.add_comment(inst, format!("{:?}", stmt));

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
                layout::Variants::Tagged { .. } => {
                    let ptr = place.place_field(fx, mir::Field::new(0));
                    let to = layout
                        .ty
                        .ty_adt_def()
                        .unwrap()
                        .discriminant_for_variant(fx.tcx, *variant_index)
                        .val;
                    let discr = CValue::const_val(fx, ptr.layout().ty, to as u64 as i64);
                    ptr.write_cvalue(fx, discr);
                }
                layout::Variants::NicheFilling {
                    dataful_variant,
                    ref niche_variants,
                    niche_start,
                    ..
                } => {
                    if *variant_index != dataful_variant {
                        let niche = place.place_field(fx, mir::Field::new(0));
                        //let niche_llty = niche.layout.immediate_llvm_type(bx.cx);
                        let niche_value = ((variant_index - *niche_variants.start()) as u128)
                            .wrapping_add(niche_start);
                        // FIXME(eddyb) Check the actual primitive type here.
                        let niche_llval = if niche_value == 0 {
                            CValue::const_val(fx, niche.layout().ty, 0)
                        } else {
                            CValue::const_val(fx, niche.layout().ty, niche_value as u64 as i64)
                        };
                        niche.write_cvalue(fx, niche_llval);
                    }
                }
            }
        }
        StatementKind::Assign(to_place, rval) => {
            let lval = trans_place(fx, to_place);
            let dest_layout = lval.layout();
            match rval {
                Rvalue::Use(operand) => {
                    let val = trans_operand(fx, operand);
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Ref(_, _, place) => {
                    let place = trans_place(fx, place);
                    let addr = place.expect_addr();
                    lval.write_cvalue(fx, CValue::ByVal(addr, dest_layout));
                }
                Rvalue::BinaryOp(bin_op, lhs, rhs) => {
                    let ty = fx.monomorphize(&lhs.ty(&fx.mir.local_decls, fx.tcx));
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = match ty.sty {
                        TypeVariants::TyBool => {
                            trans_bool_binop(fx, *bin_op, lhs, rhs, lval.layout().ty)
                        }
                        TypeVariants::TyUint(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
                        }
                        TypeVariants::TyInt(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true)
                        }
                        TypeVariants::TyFloat(_) => {
                            trans_float_binop(fx, *bin_op, lhs, rhs, lval.layout().ty)
                        }
                        TypeVariants::TyChar => {
                            trans_char_binop(fx, *bin_op, lhs, rhs, lval.layout().ty)
                        }
                        TypeVariants::TyRawPtr(..) => {
                            trans_ptr_binop(fx, *bin_op, lhs, rhs, lval.layout().ty)
                        }
                        _ => unimplemented!("binop {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
                    let ty = fx.monomorphize(&lhs.ty(&fx.mir.local_decls, fx.tcx));
                    let lhs = trans_operand(fx, lhs);
                    let rhs = trans_operand(fx, rhs);

                    let res = match ty.sty {
                        TypeVariants::TyUint(_) => {
                            trans_checked_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
                        }
                        TypeVariants::TyInt(_) => {
                            trans_checked_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true)
                        }
                        _ => unimplemented!("checked binop {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::UnaryOp(un_op, operand) => {
                    let ty = fx.monomorphize(&operand.ty(&fx.mir.local_decls, fx.tcx));
                    let layout = fx.layout_of(ty);
                    let val = trans_operand(fx, operand).load_value(fx);
                    let res = match un_op {
                        UnOp::Not => fx.bcx.ins().bnot(val),
                        UnOp::Neg => match ty.sty {
                            TypeVariants::TyInt(_) => {
                                let clif_ty = fx.cton_type(ty).unwrap();
                                let zero = fx.bcx.ins().iconst(clif_ty, 0);
                                fx.bcx.ins().isub(zero, val)
                            }
                            TypeVariants::TyFloat(_) => fx.bcx.ins().fneg(val),
                            _ => unimplemented!("un op Neg for {:?}", ty),
                        },
                    };
                    lval.write_cvalue(fx, CValue::ByVal(res, layout));
                }
                Rvalue::Cast(CastKind::ReifyFnPointer, operand, ty) => {
                    let operand = trans_operand(fx, operand);
                    let layout = fx.layout_of(ty);
                    lval.write_cvalue(fx, operand.unchecked_cast_to(layout));
                }
                Rvalue::Cast(CastKind::UnsafeFnPointer, operand, ty) => {
                    let operand = trans_operand(fx, operand);
                    let layout = fx.layout_of(ty);
                    lval.write_cvalue(fx, operand.unchecked_cast_to(layout));
                }
                Rvalue::Cast(CastKind::Misc, operand, to_ty) => {
                    let operand = trans_operand(fx, operand);
                    let from_ty = operand.layout().ty;
                    match (&from_ty.sty, &to_ty.sty) {
                        (TypeVariants::TyRef(..), TypeVariants::TyRef(..))
                        | (TypeVariants::TyRef(..), TypeVariants::TyRawPtr(..))
                        | (TypeVariants::TyRawPtr(..), TypeVariants::TyRef(..))
                        | (TypeVariants::TyRawPtr(..), TypeVariants::TyRawPtr(..)) => {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (TypeVariants::TyRawPtr(..), TypeVariants::TyUint(_))
                        | (TypeVariants::TyFnPtr(..), TypeVariants::TyUint(_))
                            if to_ty.sty == fx.tcx.types.usize.sty =>
                        {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (TypeVariants::TyUint(_), TypeVariants::TyRawPtr(..))
                            if from_ty.sty == fx.tcx.types.usize.sty =>
                        {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (TypeVariants::TyChar, TypeVariants::TyUint(_))
                        | (TypeVariants::TyUint(_), TypeVariants::TyChar)
                        | (TypeVariants::TyUint(_), TypeVariants::TyInt(_))
                        | (TypeVariants::TyUint(_), TypeVariants::TyUint(_)) => {
                            let from = operand.load_value(fx);
                            let res = crate::common::cton_intcast(
                                fx,
                                from,
                                fx.cton_type(to_ty).unwrap(),
                                false,
                            );
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyInt(_), TypeVariants::TyInt(_))
                        | (TypeVariants::TyInt(_), TypeVariants::TyUint(_)) => {
                            let from = operand.load_value(fx);
                            let res = crate::common::cton_intcast(
                                fx,
                                from,
                                fx.cton_type(to_ty).unwrap(),
                                true,
                            );
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyFloat(from_flt), TypeVariants::TyFloat(to_flt)) => {
                            let from = operand.load_value(fx);
                            let res = match (from_flt, to_flt) {
                                (FloatTy::F32, FloatTy::F64) => {
                                    fx.bcx.ins().fpromote(types::F64, from)
                                }
                                (FloatTy::F64, FloatTy::F32) => {
                                    fx.bcx.ins().fdemote(types::F32, from)
                                }
                                _ => from,
                            };
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyInt(_), TypeVariants::TyFloat(_)) => {
                            let from = operand.load_value(fx);
                            let f_type = fx.cton_type(to_ty).unwrap();
                            let res = fx.bcx.ins().fcvt_from_sint(f_type, from);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyUint(_), TypeVariants::TyFloat(_)) => {
                            let from = operand.load_value(fx);
                            let f_type = fx.cton_type(to_ty).unwrap();
                            let res = fx.bcx.ins().fcvt_from_uint(f_type, from);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyBool, TypeVariants::TyUint(_))
                        | (TypeVariants::TyBool, TypeVariants::TyInt(_)) => {
                            let to_ty = fx.cton_type(to_ty).unwrap();
                            let from = operand.load_value(fx);
                            let res = if to_ty != types::I8 {
                                fx.bcx.ins().uextend(to_ty, from)
                            } else {
                                from
                            };
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        _ => unimpl!("rval misc {:?} {:?}", from_ty, to_ty),
                    }
                }
                Rvalue::Cast(CastKind::ClosureFnPointer, operand, ty) => {
                    unimplemented!("rval closure_fn_ptr {:?} {:?}", operand, ty)
                }
                Rvalue::Cast(CastKind::Unsize, operand, ty) => {
                    unimpl!("rval unsize {:?} {:?}", operand, ty);
                }
                Rvalue::Discriminant(place) => {
                    let place = trans_place(fx, place).to_cvalue(fx);
                    let discr = trans_get_discriminant(fx, place, dest_layout);
                    lval.write_cvalue(fx, discr);
                }
                Rvalue::Repeat(operand, times) => {
                    let operand = trans_operand(fx, operand);
                    for i in 0..*times {
                        let index = fx.bcx.ins().iconst(types::I64, i as i64);
                        let to = lval.place_index(fx, index);
                        to.write_cvalue(fx, operand);
                    }
                }
                Rvalue::Len(lval) => unimpl!("rval len {:?}", lval),
                Rvalue::NullaryOp(NullOp::Box, ty) => unimplemented!("rval box {:?}", ty),
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                    assert!(
                        lval.layout()
                            .ty
                            .is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all())
                    );
                    let ty_size = fx.layout_of(ty).size.bytes();
                    let val = CValue::const_val(fx, fx.tcx.types.usize, ty_size as i64);
                    lval.write_cvalue(fx, val);
                }
                Rvalue::Aggregate(kind, operands) => match **kind {
                    AggregateKind::Array(_ty) => {
                        for (i, operand) in operands.into_iter().enumerate() {
                            let operand = trans_operand(fx, operand);
                            let index = fx.bcx.ins().iconst(types::I64, i as i64);
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
        | StatementKind::ReadForMatch(_)
        | StatementKind::Validate(_, _)
        | StatementKind::EndRegion(_)
        | StatementKind::UserAssertTy(_, _) => {}

        StatementKind::InlineAsm { .. } => unimpl!("Inline assembly is not supported"),
    }
}

pub fn trans_get_discriminant<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    value: CValue<'tcx>,
    dest_layout: TyLayout<'tcx>,
) -> CValue<'tcx> {
    let layout = value.layout();

    if layout.abi == layout::Abi::Uninhabited {
        fx.bcx.ins().trap(TrapCode::User(!0));
    }
    match layout.variants {
        layout::Variants::Single { index } => {
            let discr_val = layout.ty.ty_adt_def().map_or(index as u128, |def| {
                def.discriminant_for_variant(fx.tcx, index).val
            });
            return CValue::const_val(fx, dest_layout.ty, discr_val as u64 as i64);
        }
        layout::Variants::Tagged { .. } | layout::Variants::NicheFilling { .. } => {}
    }

    let discr = value.value_field(fx, mir::Field::new(0));
    let discr_ty = discr.layout().ty;
    let lldiscr = discr.load_value(fx);
    match layout.variants {
        layout::Variants::Single { .. } => bug!(),
        layout::Variants::Tagged { ref tag, .. } => {
            let signed = match tag.value {
                layout::Int(_, signed) => signed,
                _ => false,
            };
            let val = cton_intcast(fx, lldiscr, fx.cton_type(dest_layout.ty).unwrap(), signed);
            return CValue::ByVal(val, dest_layout);
        }
        layout::Variants::NicheFilling {
            dataful_variant,
            ref niche_variants,
            niche_start,
            ..
        } => {
            let niche_llty = fx.cton_type(discr_ty).unwrap();
            let dest_cton_ty = fx.cton_type(dest_layout.ty).unwrap();
            if niche_variants.start() == niche_variants.end() {
                let b = fx
                    .bcx
                    .ins()
                    .icmp_imm(IntCC::Equal, lldiscr, niche_start as u64 as i64);
                let if_true = fx
                    .bcx
                    .ins()
                    .iconst(dest_cton_ty, *niche_variants.start() as u64 as i64);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_cton_ty, dataful_variant as u64 as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::ByVal(val, dest_layout);
            } else {
                // Rebase from niche values to discriminant values.
                let delta = niche_start.wrapping_sub(*niche_variants.start() as u128);
                let delta = fx.bcx.ins().iconst(niche_llty, delta as u64 as i64);
                let lldiscr = fx.bcx.ins().isub(lldiscr, delta);
                let b = fx.bcx.ins().icmp_imm(
                    IntCC::UnsignedLessThanOrEqual,
                    lldiscr,
                    *niche_variants.end() as u64 as i64,
                );
                let if_true =
                    cton_intcast(fx, lldiscr, fx.cton_type(dest_layout.ty).unwrap(), false);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_cton_ty, dataful_variant as u64 as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::ByVal(val, dest_layout);
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

        // TODO HACK no encoding for icmp.i8
        use crate::common::cton_intcast;
        let (lhs, rhs) = (
            cton_intcast($fx, $lhs, types::I64, $signed),
            cton_intcast($fx, $rhs, types::I64, $signed),
        );
        let b = $fx.bcx.ins().icmp(IntCC::$cc, lhs, rhs);

        CValue::ByVal($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, fcmp($cc:ident)) => {{
        assert_eq!($fx.tcx.types.bool, $ret_ty);
        let ret_layout = $fx.layout_of($ret_ty);
        let b = $fx.bcx.ins().fcmp(FloatCC::$cc, $lhs, $rhs);
        CValue::ByVal($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, custom(|| $body:expr)) => {{
        $body
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $name:ident) => {{
        let ret_layout = $fx.layout_of($ret_ty);
        CValue::ByVal($fx.bcx.ins().$name($lhs, $rhs), ret_layout)
    }};
    (
        $fx:expr, $bin_op:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $bug_fmt:expr;
        $(
            $var:ident ($sign:pat) $name:tt $( ( $($next:tt)* ) )? ;
        )*
    ) => {{
        let lhs = $lhs.load_value($fx);
        let rhs = $rhs.load_value($fx);
        match ($bin_op, $signed) {
            $(
                (BinOp::$var, $sign) => binop_match!(@single $fx, $bug_fmt, $var, $signed, lhs, rhs, $ret_ty, $name $( ( $($next)* ) )?),
            )*
        }
    }}
}

fn trans_bool_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "bool";
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
    fx: &mut FunctionCx<'a, 'tcx>,
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
    fx: &mut FunctionCx<'a, 'tcx>,
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
            "checked int binop requires lhs and rhs of same type"
        );
    }
    let res_ty = match out_ty.sty {
        TypeVariants::TyTuple(tys) => tys[0],
        _ => bug!(
            "Checked int binop requires tuple as output, but got {:?}",
            out_ty
        ),
    };

    let res = binop_match! {
        fx, bin_op, signed, lhs, rhs, res_ty, "checked int/uint";
        Add (_) iadd;
        Sub (_) isub;
        Mul (_) imul;
        Div (_) bug;
        Rem (_) bug;
        BitXor (_) bug;
        BitAnd (_) bug;
        BitOr (_) bug;
        Shl (_) ishl;
        Shr (false) ushr;
        Shr (true) sshr;

        Eq (_) bug;
        Lt (_) bug;
        Le (_) bug;
        Ne (_) bug;
        Ge (_) bug;
        Gt (_) bug;

        Offset (_) bug;
    };

    // TODO: check for overflow
    let has_overflow = CValue::const_val(fx, fx.tcx.types.bool, 0);

    let out_place = CPlace::temp(fx, out_ty);
    out_place
        .place_field(fx, mir::Field::new(0))
        .write_cvalue(fx, res);
    println!("abc");
    out_place
        .place_field(fx, mir::Field::new(1))
        .write_cvalue(fx, has_overflow);

    out_place.to_cvalue(fx)
}

fn trans_float_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
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
                TypeVariants::TyFloat(FloatTy::F32) => fx.easy_call("fmodf", &[lhs, rhs], ty),
                TypeVariants::TyFloat(FloatTy::F64) => fx.easy_call("fmod", &[lhs, rhs], ty),
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
    fx: &mut FunctionCx<'a, 'tcx>,
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
    fx: &mut FunctionCx<'a, 'tcx>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    match lhs.layout().ty.sty {
        TypeVariants::TyRawPtr(TypeAndMut { ty, mutbl: _ }) => {
            if !ty.is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all()) {
                unimpl!("Unsized values are not yet implemented");
            }
        }
        _ => bug!("trans_ptr_binop on non ptr"),
    }
    binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "ptr";
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

        Offset (_) iadd;
    }
}

pub fn trans_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    place: &Place<'tcx>,
) -> CPlace<'tcx> {
    match place {
        Place::Local(local) => fx.get_local_place(*local),
        Place::Promoted(promoted) => crate::constant::trans_promoted(fx, promoted.0),
        Place::Static(static_) => crate::constant::codegen_static_ref(fx, static_),
        Place::Projection(projection) => {
            let base = trans_place(fx, &projection.base);
            match projection.elem {
                ProjectionElem::Deref => {
                    let layout = fx.layout_of(place.ty(&*fx.mir, fx.tcx).to_ty(fx.tcx));
                    if layout.is_unsized() {
                        unimpl!("Unsized places are not yet implemented");
                    }
                    CPlace::Addr(base.to_cvalue(fx).load_value(fx), layout)
                }
                ProjectionElem::Field(field, _ty) => base.place_field(fx, field),
                ProjectionElem::Index(local) => {
                    let index = fx.get_local_place(local).to_cvalue(fx).load_value(fx);
                    base.place_index(fx, index)
                }
                ProjectionElem::ConstantIndex {
                    offset,
                    min_length: _,
                    from_end: false,
                } => unimplemented!(
                    "projection const index {:?} offset {:?} not from end",
                    projection.base,
                    offset
                ),
                ProjectionElem::ConstantIndex {
                    offset,
                    min_length: _,
                    from_end: true,
                } => unimplemented!(
                    "projection const index {:?} offset {:?} from end",
                    projection.base,
                    offset
                ),
                ProjectionElem::Subslice { from, to } => unimplemented!(
                    "projection subslice {:?} from {} to {}",
                    projection.base,
                    from,
                    to
                ),
                ProjectionElem::Downcast(_adt_def, variant) => base.downcast_variant(fx, variant),
            }
        }
    }
}

pub fn trans_operand<'a, 'tcx>(
    fx: &mut FunctionCx<'a, 'tcx>,
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
