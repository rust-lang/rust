use rustc_mir::monomorphize::MonoItem;

use cretonne_module::{Module, Backend, FuncId, Linkage};
use cretonne_simplejit::{SimpleJITBuilder, SimpleJITBackend};

use std::any::Any;
use std::collections::HashMap;

use prelude::*;

pub fn trans_crate<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Box<Any> {
    let link_meta = ::build_link_meta(tcx.crate_hash(LOCAL_CRATE));
    let metadata = tcx.encode_metadata(&link_meta);

    let mut module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new());
    let mut context = Context::new();
    let mut def_id_fn_id_map = HashMap::new();

    {
        let mut cx = CodegenCx {
            tcx,
            module: &mut module,
            def_id_fn_id_map: &mut def_id_fn_id_map,
        };
        let cx = &mut cx;

        for mono_item in
            collector::collect_crate_mono_items(
                tcx,
                collector::MonoItemCollectionMode::Eager
            ).0 {
            match mono_item {
                MonoItem::Fn(inst) => match inst {
                    Instance {
                        def: InstanceDef::Item(def_id),
                        substs,
                    } => {
                        let sig = tcx.fn_sig(def_id);
                        let sig = cton_sig_from_fn_sig(tcx, sig, substs);
                        let func_id = {
                            let module = &mut cx.module;
                            *cx.def_id_fn_id_map.entry(inst).or_insert_with(|| {
                                module.declare_function(&tcx.absolute_item_path_str(def_id), Linkage::Local, &sig).unwrap()
                            })
                        };

                        let mut f = Function::with_name_signature(ExternalName::user(0, func_id.index() as u32), sig);

                        let mut mir = ::std::io::Cursor::new(Vec::new());
                        ::rustc_mir::util::write_mir_pretty(cx.tcx, Some(def_id), &mut mir).unwrap();
                        tcx.sess.warn(&format!("{:?}:\n\n{}", def_id, String::from_utf8_lossy(&mir.into_inner())));

                        trans_fn(cx, &mut f, inst);

                        let mut cton = String::new();
                        ::cretonne::codegen::write_function(&mut cton, &f, None).unwrap();
                        tcx.sess.warn(&cton);

                        let flags = settings::Flags::new(settings::builder());
                        match ::cretonne::codegen::verify_function(&f, &flags) {
                            Ok(_) => {}
                            Err(err) => {
                                tcx.sess.fatal(&format!("cretonne verify error: {}", err));
                            }
                        }

                        context.func = f;
                        cx.module.define_function(func_id, &mut context).unwrap();
                        context.clear();
                    }
                    _ => {}
                }
                _ => {}
            }
        }
    }

    tcx.sess.warn("Compiled everything");

    module.finalize_all();

    tcx.sess.warn("Finalized everything");

    for (inst, func_id) in def_id_fn_id_map.iter() {
        //if tcx.absolute_item_path_str(inst.def_id()) != "example::ret_42" {
        if tcx.absolute_item_path_str(inst.def_id()) != "example::option_unwrap_or" {
            continue;
        }
        let finalized_function: *const u8 = module.finalize_function(*func_id);
        /*let f: extern "C" fn(&mut u32) = unsafe { ::std::mem::transmute(finalized_function) };
        let mut res = 0u32;
        f(&mut res);
        tcx.sess.warn(&format!("ret_42 returned {}", res));*/
        let f: extern "C" fn(&mut bool, &u8, bool) = unsafe { ::std::mem::transmute(finalized_function) };
        let mut res = false;
        f(&mut res, &3, false);
        tcx.sess.warn(&format!("option_unwrap_or returned {}", res));
    }

    module.finish();

    tcx.sess.fatal("unimplemented");

    Box::new(::OngoingCodegen {
        metadata: metadata,
        //translated_module: Module::new(::cretonne_faerie::FaerieBuilder::new(,
        crate_name: tcx.crate_name(LOCAL_CRATE),
    })
}

struct CodegenCx<'a, 'tcx: 'a, B: Backend + 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &'a mut Module<B>,
    def_id_fn_id_map: &'a mut HashMap<Instance<'tcx>, FuncId>,
}

fn trans_fn<'a, 'tcx: 'a>(cx: &mut CodegenCx<'a, 'tcx, CurrentBackend>, f: &mut Function, instance: Instance<'tcx>) {
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
        def_id_fn_id_map: &mut cx.def_id_fn_id_map,
        instance,
        mir,
        bcx,
        param_substs: {
            assert!(!instance.substs.needs_infer());
            instance.substs
        },
        ebb_map,
        local_map: HashMap::new(),
    };
    let fx = &mut fx;

    let ret_param = fx.bcx.append_ebb_param(start_ebb, types::I64);
    let _ = fx.bcx.create_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: 0,
        offset: None,
    }); // Dummy stack slot for debugging

    let func_params = mir.args_iter().map(|local| {
        let layout = fx.layout_of(mir.local_decls[local].ty);
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let ty = mir.local_decls[local].ty;
        let cton_type = ::common::fixup_cton_ty(fx.cton_type(ty).unwrap_or(types::I64));
        (local, fx.bcx.append_ebb_param(start_ebb, cton_type), ty, stack_slot)
    }).collect::<Vec<(Local, Value, Ty, StackSlot)>>();

    let ret_layout = fx.layout_of(fx.instance.ty(fx.tcx).fn_sig(fx.tcx).skip_binder().output());
    fx.local_map.insert(RETURN_PLACE, CPlace::Addr(ret_param, ret_layout));

    for (local, ebb_param, ty, stack_slot) in func_params {
        let place = CPlace::from_stack_slot(fx, stack_slot, ty);
        if fx.cton_type(ty).is_some() {
            place.write_cvalue(fx, CValue::ByVal(ebb_param, place.layout()));
        } else {
            place.write_cvalue(fx, CValue::ByRef(ebb_param, place.layout()));
        }
        fx.local_map.insert(local, place);
    }

    for local in mir.vars_and_temps_iter() {
        let ty = mir.local_decls[local].ty;
        let layout = fx.layout_of(ty);
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let place = CPlace::from_stack_slot(fx, stack_slot, ty);
        fx.local_map.insert(local, place);
    }

    fx.bcx.ins().jump(*fx.ebb_map.get(&START_BLOCK).unwrap(), &[]);

    for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
        let ebb = fx.get_ebb(bb);
        fx.bcx.switch_to_block(ebb);

        for stmt in &bb_data.statements {
            trans_stmt(fx, stmt);
        }

        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                let ebb = fx.get_ebb(*target);
                fx.bcx.ins().jump(ebb, &[]);
            }
            TerminatorKind::Return => {
                fx.bcx.ins().return_(&[]);
            }
            TerminatorKind::Assert { cond, expected, msg: _, target, cleanup: _ } => {
                let cond = trans_operand(fx, cond).load_value(fx);
                let target = fx.get_ebb(*target);
                if *expected {
                    fx.bcx.ins().brz(cond, target, &[]);
                } else {
                    fx.bcx.ins().brnz(cond, target, &[]);
                }
                fx.bcx.ins().trap(TrapCode::User(!0));
            }

            TerminatorKind::SwitchInt { discr, switch_ty, values, targets } => {
                let discr = trans_operand(fx, discr).load_value(fx);
                let mut jt_data = JumpTableData::new();
                for (i, value) in values.iter().enumerate() {
                    let ebb = fx.get_ebb(targets[i]);
                    jt_data.set_entry(*value as usize, ebb);
                }
                let mut jump_table = fx.bcx.create_jump_table(jt_data);
                fx.bcx.ins().br_table(discr, jump_table);
                let otherwise_ebb = fx.get_ebb(targets[targets.len() - 1]);
                fx.bcx.ins().jump(otherwise_ebb, &[]);
            }
            TerminatorKind::Call { func, args, destination, cleanup: _ } => {
                let func_ty = func.ty(&fx.mir.local_decls, fx.tcx);
                let func = trans_operand(fx, func);
                let return_place = if let Some((place, _)) = destination {
                    trans_place(fx, place).expect_addr()
                } else {
                    fx.bcx.ins().iconst(types::I64, 0)
                };
                let args = Some(return_place)
                    .into_iter()
                    .chain(
                        args
                            .into_iter()
                            .map(|arg| {
                                let ty = arg.ty(&fx.mir.local_decls, fx.tcx);
                                let arg = trans_operand(fx, arg);
                                if let Some(_) = fx.cton_type(ty) {
                                    arg.load_value(fx)
                                } else {
                                    arg.force_stack(fx)
                                }
                            })
                    ).collect::<Vec<_>>();
                match func {
                    CValue::Func(func, _) => {
                        fx.bcx.ins().call(func, &args);
                    }
                    func => {
                        let func = func.load_value(fx);
                        let sig = match func_ty.sty {
                            TypeVariants::TyFnDef(def_id, _substs) => fx.tcx.fn_sig(def_id),
                            TypeVariants::TyFnPtr(fn_sig) => fn_sig,
                            _ => bug!("Calling non function type {:?}", func_ty),
                        };
                        let sig = fx.bcx.import_signature(cton_sig_from_fn_sig(fx.tcx, sig, fx.param_substs));
                        fx.bcx.ins().call_indirect(sig, func, &args);
                    }
                }
                if let Some((_, dest)) = *destination {
                    let ret_ebb = fx.get_ebb(dest);
                    fx.bcx.ins().jump(ret_ebb, &[]);
                } else {
                    fx.bcx.ins().trap(TrapCode::User(!0));
                }
            }
            TerminatorKind::Resume | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                fx.bcx.ins().trap(TrapCode::User(!0));
            }
            TerminatorKind::Yield { .. } |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => {
                bug!("shouldn't exist at trans {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop { .. } | TerminatorKind::DropAndReplace { .. } | TerminatorKind::GeneratorDrop { .. } => {
                unimplemented!("terminator {:?}", bb_data.terminator());
            }
        }
    }

    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
}

fn trans_stmt<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, stmt: &Statement<'tcx>) {
    match &stmt.kind {
        StatementKind::SetDiscriminant { place, variant_index } => {
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
                    let to = layout.ty.ty_adt_def().unwrap()
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
                    let lhs = trans_operand(fx, lhs).load_value(fx);
                    let rhs = trans_operand(fx, rhs).load_value(fx);

                    let res = match ty.sty {
                        TypeVariants::TyUint(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, ty, false, false)
                        }
                        TypeVariants::TyInt(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, ty, true, false)
                        }
                        _ => unimplemented!("bin op {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                }
                Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
                    let ty = fx.monomorphize(&lhs.ty(&fx.mir.local_decls, fx.tcx));
                    let lhs = trans_operand(fx, lhs).load_value(fx);
                    let rhs = trans_operand(fx, rhs).load_value(fx);

                    let res = match ty.sty {
                        TypeVariants::TyUint(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, ty, false, true)
                        }
                        TypeVariants::TyInt(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, ty, true, true)
                        }
                        _ => unimplemented!("checked bin op {:?} for {:?}", bin_op, ty),
                    };
                    lval.write_cvalue(fx, res);
                    unimplemented!("checked bin op {:?}", bin_op);
                }
                Rvalue::UnaryOp(un_op, operand) => {
                    let ty = fx.monomorphize(&operand.ty(&fx.mir.local_decls, fx.tcx));
                    let layout = fx.layout_of(ty);
                    let val = trans_operand(fx, operand).load_value(fx);
                    let res = match un_op {
                        UnOp::Not => fx.bcx.ins().bnot(val),
                        UnOp::Neg => match ty.sty {
                            TypeVariants::TyFloat(_) => fx.bcx.ins().fneg(val),
                            _ => unimplemented!("un op Neg for {:?}", ty),
                        }
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
                Rvalue::Discriminant(place) => {
                    let place = trans_place(fx, place);
                    let dest_cton_ty = fx.cton_type(dest_layout.ty).unwrap();
                    let layout = lval.layout();

                    if layout.abi == layout::Abi::Uninhabited {
                        fx.bcx.ins().trap(TrapCode::User(!0));
                    }
                    match layout.variants {
                        layout::Variants::Single { index } => {
                            let discr_val = layout.ty.ty_adt_def().map_or(
                                index as u128,
                                |def| def.discriminant_for_variant(fx.tcx, index).val);
                            let val = CValue::const_val(fx, dest_layout.ty, discr_val as u64 as i64);
                            lval.write_cvalue(fx, val);
                            return;
                        }
                        layout::Variants::Tagged { .. } |
                        layout::Variants::NicheFilling { .. } => {},
                    }

                    let discr = place.to_cvalue(fx).value_field(fx, mir::Field::new(0));
                    let discr_ty = discr.layout().ty;
                    let lldiscr = discr.load_value(fx);
                    match layout.variants {
                        layout::Variants::Single { .. } => bug!(),
                        layout::Variants::Tagged { ref tag, .. } => {
                            let signed = match tag.value {
                                layout::Int(_, signed) => signed,
                                _ => false
                            };
                            let val = cton_intcast(fx, lldiscr, discr_ty, dest_layout.ty, signed);
                            lval.write_cvalue(fx, CValue::ByVal(val, dest_layout));
                        }
                        layout::Variants::NicheFilling {
                            dataful_variant,
                            ref niche_variants,
                            niche_start,
                            ..
                        } => {
                            let niche_llty = fx.cton_type(discr_ty).unwrap();
                            if niche_variants.start() == niche_variants.end() {
                                let b = fx.bcx.ins().icmp_imm(IntCC::Equal, lldiscr, niche_start as u64 as i64);
                                let if_true = fx.bcx.ins().iconst(dest_cton_ty, *niche_variants.start() as u64 as i64);
                                let if_false = fx.bcx.ins().iconst(dest_cton_ty, dataful_variant as u64 as i64);
                                let val = fx.bcx.ins().select(b, if_true, if_false);
                                lval.write_cvalue(fx, CValue::ByVal(val, dest_layout));
                            } else {
                                // Rebase from niche values to discriminant values.
                                let delta = niche_start.wrapping_sub(*niche_variants.start() as u128);
                                let delta = fx.bcx.ins().iconst(niche_llty, delta as u64 as i64);
                                let lldiscr = fx.bcx.ins().isub(lldiscr, delta);
                                let b = fx.bcx.ins().icmp_imm(IntCC::UnsignedLessThanOrEqual, lldiscr, *niche_variants.end() as u64 as i64);
                                let if_true = cton_intcast(fx, lldiscr, discr_ty, dest_layout.ty, false);
                                let if_false = fx.bcx.ins().iconst(niche_llty, dataful_variant as u64 as i64);
                                let val = fx.bcx.ins().select(b, if_true, if_false);
                                lval.write_cvalue(fx, CValue::ByVal(val, dest_layout));
                            }
                        }
                    }
                }
                rval => unimplemented!("rval {:?}", rval),
            }
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {}
        _ => unimplemented!("stmt {:?}", stmt),
    }
}

fn trans_int_binop<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, bin_op: BinOp, lhs: Value, rhs: Value, ty: Ty<'tcx>, signed: bool, checked: bool) -> CValue<'tcx> {
    let res = match (bin_op, signed) {
        (BinOp::Add, _) => fx.bcx.ins().iadd(lhs, rhs),
        (BinOp::Sub, _) => fx.bcx.ins().isub(lhs, rhs),
        (BinOp::Mul, _) => fx.bcx.ins().imul(lhs, rhs),
        (BinOp::Div, false) => fx.bcx.ins().udiv(lhs, rhs),
        (BinOp::Div, true) => fx.bcx.ins().sdiv(lhs, rhs),
        (BinOp::Rem, false) => fx.bcx.ins().urem(lhs, rhs),
        (BinOp::Rem, true) => fx.bcx.ins().srem(lhs, rhs),
        (BinOp::BitXor, _) => fx.bcx.ins().bxor(lhs, rhs),
        (BinOp::BitAnd, _) => fx.bcx.ins().band(lhs, rhs),
        (BinOp::BitOr, _) => fx.bcx.ins().bor(lhs, rhs),
        (BinOp::Shl, _) => fx.bcx.ins().ishl(lhs, rhs),
        (BinOp::Shr, false) => fx.bcx.ins().ushr(lhs, rhs),
        (BinOp::Shr, true) => fx.bcx.ins().sshr(lhs, rhs),
        (BinOp::Eq, _) => fx.bcx.ins().icmp(IntCC::Equal , lhs, rhs),
        (BinOp::Lt, false) => fx.bcx.ins().icmp(IntCC::UnsignedLessThan , lhs, rhs),
        (BinOp::Lt, true) => fx.bcx.ins().icmp(IntCC::SignedLessThan , lhs, rhs),
        (BinOp::Le, false) => fx.bcx.ins().icmp(IntCC::UnsignedLessThanOrEqual , lhs, rhs),
        (BinOp::Le, true) => fx.bcx.ins().icmp(IntCC::SignedLessThanOrEqual , lhs, rhs),
        (BinOp::Ne, _) => fx.bcx.ins().icmp(IntCC::NotEqual , lhs, rhs),
        (BinOp::Ge, false) => fx.bcx.ins().icmp(IntCC::UnsignedGreaterThanOrEqual , lhs, rhs),
        (BinOp::Ge, true) => fx.bcx.ins().icmp(IntCC::SignedGreaterThanOrEqual , lhs, rhs),
        (BinOp::Gt, false) => fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan , lhs, rhs),
        (BinOp::Gt, true) => fx.bcx.ins().icmp(IntCC::SignedGreaterThan , lhs, rhs),
        (BinOp::Offset, _) => bug!("bin op Offset on non ptr lhs: {:?} rhs: {:?}", lhs, rhs),
    };
    // TODO: return correct value for checked binops
    CValue::ByVal(res, fx.layout_of(ty))
}

fn trans_place<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, place: &Place<'tcx>) -> CPlace<'tcx> {
    match place {
        Place::Local(local) => fx.get_local_place(*local),
        Place::Projection(projection) => {
            let base = trans_place(fx, &projection.base);
            match projection.elem {
                ProjectionElem::Deref => {
                    CPlace::Addr(base.to_cvalue(fx).load_value(fx), fx.layout_of(place.ty(&*fx.mir, fx.tcx).to_ty(fx.tcx)))
                }
                ProjectionElem::Field(field, ty) => {
                    base.place_field(fx, field)
                }
                ProjectionElem::Downcast(adt_def, variant) => {
                    base.downcast_variant(fx, variant)
                }
                _ => unimplemented!("projection {:?}", projection),
            }
        }
        place => unimplemented!("place {:?}", place),
    }
}

fn trans_operand<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx>, operand: &Operand<'tcx>) -> CValue<'tcx> {
    match operand {
        Operand::Move(place) |
        Operand::Copy(place) => {
            let cplace = trans_place(fx, place);
            cplace.to_cvalue(fx)
        },
        Operand::Constant(const_) => {
            match const_.literal {
                Literal::Value { value } => {
                    let layout = fx.layout_of(const_.ty);
                    match const_.ty.sty {
                        TypeVariants::TyBool => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            CValue::const_val(fx, const_.ty, bits as u64 as i64)
                        }
                        TypeVariants::TyUint(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            CValue::const_val(fx, const_.ty, bits as u64 as i64)
                        }
                        TypeVariants::TyInt(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            CValue::const_val(fx, const_.ty, bits as i128 as i64)
                        }
                        TypeVariants::TyFnDef(def_id, substs) => {
                            let func_ref = fx.get_function_ref(Instance::new(def_id, substs));
                            CValue::Func(func_ref, fx.layout_of(const_.ty))
                        }
                        _ => unimplemented!("value {:?} ty {:?}", value, const_.ty),
                    }
                }
                _ => unimplemented!()
            }
        }
    }
}
