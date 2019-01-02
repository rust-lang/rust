use crate::prelude::*;

struct PrintOnPanic<F: Fn() -> String>(F);
impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            println!("{}", (self.0)());
        }
    }
}

pub fn trans_mono_item<'a, 'clif, 'tcx: 'a, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'a, 'clif, 'tcx, B>,
    mono_item: MonoItem<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;
    match mono_item {
        MonoItem::Fn(inst) => {
            let _inst_guard =
                PrintOnPanic(|| format!("{:?} {}", inst, tcx.symbol_name(inst).as_str()));
            debug_assert!(!inst.substs.needs_infer());
            let _mir_guard = PrintOnPanic(|| {
                match inst.def {
                    InstanceDef::Item(_)
                    | InstanceDef::DropGlue(_, _)
                    | InstanceDef::Virtual(_, _)
                        if inst.def_id().krate == LOCAL_CRATE =>
                    {
                        let mut mir = ::std::io::Cursor::new(Vec::new());
                        crate::rustc_mir::util::write_mir_pretty(
                            tcx,
                            Some(inst.def_id()),
                            &mut mir,
                        )
                        .unwrap();
                        String::from_utf8(mir.into_inner()).unwrap()
                    }
                    _ => {
                        // FIXME fix write_mir_pretty for these instances
                        format!("{:#?}", tcx.instance_mir(inst.def))
                    }
                }
            });

            trans_fn(cx, inst, linkage);
        }
        MonoItem::Static(def_id) => {
            crate::constant::codegen_static(&mut cx.ccx, def_id);
        }
        MonoItem::GlobalAsm(node_id) => tcx
            .sess
            .fatal(&format!("Unimplemented global asm mono item {:?}", node_id)),
    }
}

fn trans_fn<'a, 'clif, 'tcx: 'a, B: Backend + 'static>(
    cx: &mut crate::CodegenCx<'a, 'clif, 'tcx, B>,
    instance: Instance<'tcx>,
    linkage: Linkage,
) {
    let tcx = cx.tcx;

    // Step 1. Get mir
    let mir = tcx.instance_mir(instance.def);

    // Step 2. Declare function
    let (name, sig) = get_function_name_and_sig(tcx, instance);
    let func_id = cx.module
        .declare_function(&name, linkage, &sig)
        .unwrap();

    // Step 3. Make FunctionBuilder
    let mut func = Function::with_name_signature(ExternalName::user(0, 0), sig);
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Step 4. Predefine ebb's
    let start_ebb = bcx.create_ebb();
    let mut ebb_map: HashMap<BasicBlock, Ebb> = HashMap::new();
    for (bb, _bb_data) in mir.basic_blocks().iter_enumerated() {
        ebb_map.insert(bb, bcx.create_ebb());
    }

    // Step 5. Make FunctionCx
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
    };

    // Step 6. Codegen function
    with_unimpl_span(fx.mir.span, || {
        crate::abi::codegen_fn_prelude(&mut fx, start_ebb);
        codegen_fn_content(&mut fx);
    });

    // Step 7. Write function to file for debugging
    #[cfg(debug_assertions)]
    fx.write_clif_file();

    // Step 8. Verify function
    verify_func(tcx, fx.clif_comments, &func);

    // Step 9. Define function
    cx.caches.context.func = func;
    cx.module
        .define_function(func_id, &mut cx.caches.context)
        .unwrap();
    cx.caches.context.clear();
}

fn verify_func(tcx: TyCtxt, writer: crate::pretty_clif::CommentWriter, func: &Function) {
    let flags = settings::Flags::new(settings::builder());
    match ::cranelift::codegen::verify_function(&func, &flags) {
        Ok(_) => {}
        Err(err) => {
            tcx.sess.err(&format!("{:?}", err));
            let pretty_error = ::cranelift::codegen::print_errors::pretty_verifier_error(
                &func,
                None,
                Some(Box::new(&writer)),
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
                let cond = trans_operand(fx, cond).load_scalar(fx);
                // TODO HACK brz/brnz for i8/i16 is not yet implemented
                let cond = fx.bcx.ins().uextend(types::I32, cond);
                let target = fx.get_ebb(*target);
                if *expected {
                    fx.bcx.ins().brnz(cond, target, &[]);
                } else {
                    fx.bcx.ins().brz(cond, target, &[]);
                };
                trap_panic(&mut fx.bcx);
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
            TerminatorKind::Resume | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                trap_unreachable(&mut fx.bcx);
            }
            TerminatorKind::Yield { .. }
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::DropAndReplace { .. } => {
                bug!("shouldn't exist at trans {:?}", bb_data.terminator());
            }
            TerminatorKind::Drop {
                location,
                target,
                unwind: _,
            } => {
                let ty = location.ty(fx.mir, fx.tcx).to_ty(fx.tcx);
                let ty = fx.monomorphize(&ty);
                let drop_fn = crate::rustc_mir::monomorphize::resolve_drop_in_place(fx.tcx, ty);

                if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
                    // we don't actually need to drop anything
                } else {
                    let drop_place = trans_place(fx, location);
                    let arg_place = CPlace::temp(
                        fx,
                        fx.tcx.mk_ref(
                            &ty::RegionKind::ReErased,
                            TypeAndMut {
                                ty,
                                mutbl: crate::rustc::hir::Mutability::MutMutable,
                            },
                        ),
                    );
                    drop_place.write_place_ref(fx, arg_place);
                    match ty.sty {
                        ty::Dynamic(..) => {
                            fx.tcx.sess.warn("Drop for trait object");
                        }
                        _ => {
                            let drop_fn_ty = drop_fn.ty(fx.tcx);
                            let arg_value = arg_place.to_cvalue(fx);
                            crate::abi::codegen_call_inner(
                                fx,
                                None,
                                drop_fn_ty,
                                vec![arg_value],
                                None,
                            );
                        }
                    }
                    /*
                    let (args1, args2);
                    /*let mut args = if let Some(llextra) = place.llextra {
                        args2 = [place.llval, llextra];
                        &args2[..]
                    } else {
                        args1 = [place.llval];
                        &args1[..]
                    };*/
                    let (drop_fn, fn_ty) = match ty.sty {
                    ty::Dynamic(..) => {
                    let fn_ty = drop_fn.ty(bx.cx.tcx);
                    let sig = common::ty_fn_sig(bx.cx, fn_ty);
                    let sig = bx.tcx().normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                    );
                    let fn_ty = FnType::new_vtable(bx.cx, sig, &[]);
                    let vtable = args[1];
                    args = &args[..1];
                    (meth::DESTRUCTOR.get_fn(&bx, vtable, &fn_ty), fn_ty)
                    }
                    _ => {
                    let value = place.to_cvalue(fx);
                    (callee::get_fn(bx.cx, drop_fn),
                    FnType::of_instance(bx.cx, &drop_fn))
                    }
                    };
                    do_call(self, bx, fn_ty, drop_fn, args,
                    Some((ReturnDest::Nothing, target)),
                    unwind);*/
                }

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
}

fn trans_stmt<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    cur_ebb: Ebb,
    stmt: &Statement<'tcx>,
) {
    let _print_guard = PrintOnPanic(|| format!("stmt {:?}", stmt));

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
                        let niche_value =
                            ((variant_index.as_u32() - niche_variants.start().as_u32()) as u128)
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
                        ty::Bool => trans_bool_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::Uint(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
                        }
                        ty::Int(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true)
                        }
                        ty::Float(_) => trans_float_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::Char => trans_char_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
                        ty::RawPtr(..) => trans_ptr_binop(fx, *bin_op, lhs, rhs, lval.layout().ty),
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
                                ty::Uint(_) | ty::Int(_) => fx.bcx.ins().bnot(val),
                                _ => unimplemented!("un op Not for {:?}", layout.ty),
                            }
                        }
                        UnOp::Neg => match layout.ty.sty {
                            ty::Int(_) => {
                                let clif_ty = fx.clif_type(layout.ty).unwrap();
                                let zero = fx.bcx.ins().iconst(clif_ty, 0);
                                fx.bcx.ins().isub(zero, val)
                            }
                            ty::Float(_) => fx.bcx.ins().fneg(val),
                            _ => unimplemented!("un op Neg for {:?}", layout.ty),
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
                        (ty::Ref(..), ty::Ref(..))
                        | (ty::Ref(..), ty::RawPtr(..))
                        | (ty::RawPtr(..), ty::Ref(..))
                        | (ty::RawPtr(..), ty::RawPtr(..))
                        | (ty::FnPtr(..), ty::RawPtr(..)) => {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (ty::RawPtr(..), ty::Uint(_))
                        | (ty::RawPtr(..), ty::Int(_))
                        | (ty::FnPtr(..), ty::Uint(_))
                            if to_ty.sty == fx.tcx.types.usize.sty
                                || to_ty.sty == fx.tcx.types.isize.sty
                                || fx.clif_type(to_ty).unwrap() == pointer_ty(fx.tcx) =>
                        {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (ty::Uint(_), ty::RawPtr(..)) if from_ty.sty == fx.tcx.types.usize.sty => {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (ty::Char, ty::Uint(_))
                        | (ty::Uint(_), ty::Char)
                        | (ty::Uint(_), ty::Int(_))
                        | (ty::Uint(_), ty::Uint(_)) => {
                            let from = operand.load_scalar(fx);
                            let res = crate::common::clif_intcast(
                                fx,
                                from,
                                fx.clif_type(to_ty).unwrap(),
                                false,
                            );
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (ty::Int(_), ty::Int(_)) | (ty::Int(_), ty::Uint(_)) => {
                            let from = operand.load_scalar(fx);
                            let res = crate::common::clif_intcast(
                                fx,
                                from,
                                fx.clif_type(to_ty).unwrap(),
                                true,
                            );
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (ty::Float(from_flt), ty::Float(to_flt)) => {
                            let from = operand.load_scalar(fx);
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
                        (ty::Int(_), ty::Float(_)) => {
                            let from_ty = fx.clif_type(from_ty).unwrap();
                            let from = operand.load_scalar(fx);
                            // FIXME missing encoding for fcvt_from_sint.f32.i8
                            let from = if from_ty == types::I8 || from_ty == types::I16 {
                                fx.bcx.ins().sextend(types::I32, from)
                            } else {
                                from
                            };
                            let f_type = fx.clif_type(to_ty).unwrap();
                            let res = fx.bcx.ins().fcvt_from_sint(f_type, from);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (ty::Uint(_), ty::Float(_)) => {
                            let from_ty = fx.clif_type(from_ty).unwrap();
                            let from = operand.load_scalar(fx);
                            // FIXME missing encoding for fcvt_from_uint.f32.i8
                            let from = if from_ty == types::I8 || from_ty == types::I16 {
                                fx.bcx.ins().uextend(types::I32, from)
                            } else {
                                from
                            };
                            let f_type = fx.clif_type(to_ty).unwrap();
                            let res = fx.bcx.ins().fcvt_from_uint(f_type, from);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (ty::Bool, ty::Uint(_)) | (ty::Bool, ty::Int(_)) => {
                            let to_ty = fx.clif_type(to_ty).unwrap();
                            let from = operand.load_scalar(fx);
                            let res = if to_ty != types::I8 {
                                fx.bcx.ins().uextend(to_ty, from)
                            } else {
                                from
                            };
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (ty::Adt(adt_def, _substs), ty::Uint(_)) | (ty::Adt(adt_def, _substs), ty::Int(_)) if adt_def.is_enum() => {
                            let discr = trans_get_discriminant(fx, operand, fx.layout_of(to_ty));
                            lval.write_cvalue(fx, discr);
                        }
                        _ => unimpl!("rval misc {:?} {:?}", from_ty, to_ty),
                    }
                }
                Rvalue::Cast(CastKind::ClosureFnPointer, operand, ty) => {
                    unimplemented!("rval closure_fn_ptr {:?} {:?}", operand, ty)
                }
                Rvalue::Cast(CastKind::Unsize, operand, _ty) => {
                    let operand = trans_operand(fx, operand);
                    operand.unsize_value(fx, lval);
                }
                Rvalue::Discriminant(place) => {
                    let place = trans_place(fx, place).to_cvalue(fx);
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
                    lval.write_cvalue(fx, CValue::ByVal(len, usize_layout));
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
                    lval.write_cvalue(fx, CValue::ByVal(ptr, box_layout));
                }
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                    assert!(lval
                        .layout()
                        .ty
                        .is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all()));
                    let ty_size = fx.layout_of(ty).size.bytes();
                    let val = CValue::const_val(fx, fx.tcx.types.usize, ty_size as i64);
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

        StatementKind::InlineAsm { .. } => unimpl!("Inline assembly is not supported"),
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
        ty::Slice(_elem_ty) => match place {
            CPlace::Addr(_, size, _) => size.unwrap(),
            CPlace::Var(_, _) => unreachable!(),
        },
        _ => bug!("Rvalue::Len({:?})", place),
    }
}

pub fn trans_get_discriminant<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    value: CValue<'tcx>,
    dest_layout: TyLayout<'tcx>,
) -> CValue<'tcx> {
    let layout = value.layout();

    if layout.abi == layout::Abi::Uninhabited {
        trap_unreachable(&mut fx.bcx);
    }
    match layout.variants {
        layout::Variants::Single { index } => {
            let discr_val = layout
                .ty
                .ty_adt_def()
                .map_or(index.as_u32() as u128, |def| {
                    def.discriminant_for_variant(fx.tcx, index).val
                });
            return CValue::const_val(fx, dest_layout.ty, discr_val as u64 as i64);
        }
        layout::Variants::Tagged { .. } | layout::Variants::NicheFilling { .. } => {}
    }

    let discr = value.value_field(fx, mir::Field::new(0));
    let discr_ty = discr.layout().ty;
    let lldiscr = discr.load_scalar(fx);
    match layout.variants {
        layout::Variants::Single { .. } => bug!(),
        layout::Variants::Tagged { ref tag, .. } => {
            let signed = match tag.value {
                layout::Int(_, signed) => signed,
                _ => false,
            };
            let val = clif_intcast(fx, lldiscr, fx.clif_type(dest_layout.ty).unwrap(), signed);
            return CValue::ByVal(val, dest_layout);
        }
        layout::Variants::NicheFilling {
            dataful_variant,
            ref niche_variants,
            niche_start,
            ..
        } => {
            let niche_llty = fx.clif_type(discr_ty).unwrap();
            let dest_clif_ty = fx.clif_type(dest_layout.ty).unwrap();
            if niche_variants.start() == niche_variants.end() {
                let b = fx
                    .bcx
                    .ins()
                    .icmp_imm(IntCC::Equal, lldiscr, niche_start as u64 as i64);
                let if_true = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, niche_variants.start().as_u32() as i64);
                let if_false = fx
                    .bcx
                    .ins()
                    .iconst(dest_clif_ty, dataful_variant.as_u32() as i64);
                let val = fx.bcx.ins().select(b, if_true, if_false);
                return CValue::ByVal(val, dest_layout);
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

        let b = $fx.bcx.ins().icmp(IntCC::$cc, $lhs, $rhs);
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
    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            in_lhs.layout().ty,
            in_rhs.layout().ty,
            "checked int binop requires lhs and rhs of same type"
        );
    }

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);
    let res = match bin_op {
        BinOp::Add => fx.bcx.ins().iadd(lhs, rhs),
        BinOp::Sub => fx.bcx.ins().isub(lhs, rhs),
        BinOp::Mul => fx.bcx.ins().imul(lhs, rhs),
        BinOp::Shl => fx.bcx.ins().ishl(lhs, rhs),
        BinOp::Shr => {
            if !signed {
                fx.bcx.ins().ushr(lhs, rhs)
            } else {
                fx.bcx.ins().sshr(lhs, rhs)
            }
        }
        _ => bug!(
            "binop {:?} on checked int/uint lhs: {:?} rhs: {:?}",
            bin_op,
            in_lhs,
            in_rhs
        ),
    };

    // TODO: check for overflow
    let has_overflow = fx.bcx.ins().iconst(types::I8, 0);

    let out_place = CPlace::temp(fx, out_ty);
    let out_layout = out_place.layout();
    out_place.write_cvalue(fx, CValue::ByValPair(res, has_overflow, out_layout));

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
    match lhs.layout().ty.sty {
        ty::RawPtr(TypeAndMut { ty, mutbl: _ }) => {
            if ty.is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all()) {
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

                    Offset (_) iadd;
                }
            } else {
                let lhs = lhs.load_value_pair(fx).0;
                let rhs = rhs.load_value_pair(fx).0;
                let res = match bin_op {
                    BinOp::Eq => fx.bcx.ins().icmp(IntCC::Equal, lhs, rhs),
                    BinOp::Ne => fx.bcx.ins().icmp(IntCC::NotEqual, lhs, rhs),
                    _ => unimplemented!(
                        "trans_ptr_binop({:?}, <fat ptr>, <fat ptr>) not implemented",
                        bin_op
                    ),
                };

                assert_eq!(fx.tcx.types.bool, ret_ty);
                let ret_layout = fx.layout_of(ret_ty);
                CValue::ByVal(fx.bcx.ins().bint(types::I8, res), ret_layout)
            }
        }
        _ => bug!("trans_ptr_binop on non ptr"),
    }
}

pub fn trans_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    place: &Place<'tcx>,
) -> CPlace<'tcx> {
    match place {
        Place::Local(local) => fx.get_local_place(*local),
        Place::Promoted(promoted) => crate::constant::trans_promoted(fx, promoted.0),
        Place::Static(static_) => crate::constant::codegen_static_ref(fx, static_),
        Place::Projection(projection) => {
            let base = trans_place(fx, &projection.base);
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
                ProjectionElem::Subslice { from, to } => unimpl!(
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
