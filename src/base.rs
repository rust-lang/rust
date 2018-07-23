use prelude::*;

pub fn trans_mono_item<'a, 'tcx: 'a>(cx: &mut CodegenCx<'a, 'tcx, CurrentBackend>, context: &mut Context, mono_item: MonoItem<'tcx>) {
    let tcx = cx.tcx;

    match mono_item {
        MonoItem::Fn(inst) => match inst {
            Instance {
                def: InstanceDef::Item(def_id),
                substs,
            } => {
                let mut mir = ::std::io::Cursor::new(Vec::new());
                ::rustc_mir::util::write_mir_pretty(tcx, Some(def_id), &mut mir).unwrap();
                tcx.sess.warn(&format!("{:?}:\n\n{}", def_id, String::from_utf8_lossy(&mir.into_inner())));

                let fn_ty = inst.ty(tcx);
                let fn_ty = tcx.subst_and_normalize_erasing_regions(
                    substs,
                    ty::ParamEnv::reveal_all(),
                    &fn_ty,
                );
                let sig = cton_sig_from_fn_ty(tcx, fn_ty);
                let func_id = {
                    let module = &mut cx.module;
                    *cx.def_id_fn_id_map.entry(inst).or_insert_with(|| {
                        let def_path_based_names = ::rustc_mir::monomorphize::item::DefPathBasedNames::new(tcx, false, false);
                        let mut name = String::new();
                        def_path_based_names.push_instance_as_string(inst, &mut name);
                        module.declare_function(&name, Linkage::Local, &sig).unwrap()
                    })
                };

                let mut f = Function::with_name_signature(ExternalName::user(0, func_id.index() as u32), sig);

                let comments = match ::base::trans_fn(cx, &mut f, inst){
                    Ok(comments) => comments,
                    Err(err) => {
                        tcx.sess.err(&err);
                        return;
                    }
                };

                let mut writer = ::pretty_clif::CommentWriter(comments);
                let mut cton = String::new();
                ::cranelift::codegen::write::decorate_function(&mut writer, &mut cton, &f, None).unwrap();
                tcx.sess.warn(&cton);

                let flags = settings::Flags::new(settings::builder());
                match ::cranelift::codegen::verify_function(&f, &flags) {
                    Ok(_) => {}
                    Err(err) => {
                        tcx.sess.err(&format!("{:?}", err));
                        let pretty_error = ::cranelift::codegen::print_errors::pretty_verifier_error(&f, None, Some(Box::new(writer)), &err);
                        tcx.sess.fatal(&format!("cretonne verify error:\n{}", pretty_error));
                    }
                }

                context.func = f;
                // TODO: cranelift doesn't yet support some of the things needed
                // cx.module.define_function(func_id, context).unwrap();

                context.clear();
            }
            inst => cx.tcx.sess.warn(&format!("Unimplemented instance {:?}", inst)),
        }
        MonoItem::Static(def_id) => cx.tcx.sess.err(&format!("Unimplemented static mono item {:?}", def_id)),
        MonoItem::GlobalAsm(node_id) => cx.tcx.sess.err(&format!("Unimplemented global asm mono item {:?}", node_id)),
    }
}

pub fn trans_fn<'a, 'tcx: 'a>(cx: &mut CodegenCx<'a, 'tcx, CurrentBackend>, f: &mut Function, instance: Instance<'tcx>) -> Result<HashMap<Inst, String>, String> {
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
        comments: HashMap::new(),
        constants: &mut cx.constants,
    };
    let fx = &mut fx;

    ::abi::codegen_fn_prelude(fx, start_ebb);

    fx.bcx.ins().jump(*fx.ebb_map.get(&START_BLOCK).unwrap(), &[]);

    for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
        let ebb = fx.get_ebb(bb);
        fx.bcx.switch_to_block(ebb);

        fx.bcx.ins().nop();
        for stmt in &bb_data.statements {
            trans_stmt(fx, ebb, stmt)?;
        }

        let mut terminator_head = "\n".to_string();
        bb_data.terminator().kind.fmt_head(&mut terminator_head).unwrap();
        let inst = fx.bcx.func.layout.last_inst(ebb).unwrap();
        fx.add_comment(inst, terminator_head);

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
                };
                fx.bcx.ins().trap(TrapCode::User(!0));
            }

            TerminatorKind::SwitchInt { discr, switch_ty: _, values, targets } => {
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
                ::abi::codegen_call(fx, func, args, destination);
            }
            TerminatorKind::Resume | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                fx.bcx.ins().trap(TrapCode::User(!0));
            }
            TerminatorKind::Yield { .. } |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => {
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

    Ok(fx.comments.clone())
}

fn trans_stmt<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, cur_ebb: Ebb, stmt: &Statement<'tcx>) -> Result<(), String> {
    fx.tcx.sess.warn(&format!("stmt {:?}", stmt));

    let inst = fx.bcx.func.layout.last_inst(cur_ebb).unwrap();
    fx.add_comment(inst, format!("{:?}", stmt));

    match &stmt.kind {
        StatementKind::SetDiscriminant { place, variant_index } => {
            let place = trans_place(fx, place);
            let layout = place.layout();
            if layout.for_variant(&*fx, *variant_index).abi == layout::Abi::Uninhabited {
                return Ok(());
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
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false, false)
                        }
                        TypeVariants::TyInt(_) => {
                            trans_int_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, true, false)
                        }
                        TypeVariants::TyRawPtr(..) => {
                            trans_ptr_binop(fx, *bin_op, lhs, rhs, lval.layout().ty, false)
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
                    return Err(format!("checked bin op {:?}", bin_op));
                    lval.write_cvalue(fx, res);
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
                Rvalue::Cast(CastKind::Misc, operand, to_ty) => {
                    let operand = trans_operand(fx, operand);
                    let from_ty = operand.layout().ty;
                    match (&from_ty.sty, &to_ty.sty) {
                        (TypeVariants::TyRef(..), TypeVariants::TyRef(..)) |
                        (TypeVariants::TyRef(..), TypeVariants::TyRawPtr(..)) |
                        (TypeVariants::TyRawPtr(..), TypeVariants::TyRef(..)) |
                        (TypeVariants::TyRawPtr(..), TypeVariants::TyRawPtr(..)) => {
                            lval.write_cvalue(fx, operand.unchecked_cast_to(dest_layout));
                        }
                        (TypeVariants::TyChar, TypeVariants::TyUint(_)) |
                        (TypeVariants::TyUint(_), TypeVariants::TyInt(_)) |
                        (TypeVariants::TyUint(_), TypeVariants::TyUint(_)) => {
                            let from = operand.load_value(fx);
                            let res = ::common::cton_intcast(fx, from, from_ty, to_ty, false);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        (TypeVariants::TyInt(_), TypeVariants::TyInt(_)) |
                        (TypeVariants::TyInt(_), TypeVariants::TyUint(_)) => {
                            let from = operand.load_value(fx);
                            let res = ::common::cton_intcast(fx, from, from_ty, to_ty, true);
                            lval.write_cvalue(fx, CValue::ByVal(res, dest_layout));
                        }
                        _ => return Err(format!("rval misc {:?} {:?}", operand, to_ty)),
                    }
                },
                Rvalue::Cast(CastKind::ClosureFnPointer, operand, ty) => unimplemented!("rval closure_fn_ptr {:?} {:?}", operand, ty),
                Rvalue::Cast(CastKind::Unsize, operand, ty) => return Err(format!("rval unsize {:?} {:?}", operand, ty)),
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
                            return Ok(());
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
                Rvalue::Repeat(operand, times) => unimplemented!("rval repeat {:?} {:?}", operand, times),
                Rvalue::Len(lval) => return Err(format!("rval len {:?}", lval)),
                Rvalue::NullaryOp(NullOp::Box, ty) => unimplemented!("rval box {:?}", ty),
                Rvalue::NullaryOp(NullOp::SizeOf, ty) => unimplemented!("rval size_of {:?}", ty),
                Rvalue::Aggregate(_, _) => bug!("shouldn't exist at trans {:?}", rval),
            }
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop | StatementKind::ReadForMatch(_) | StatementKind::Validate(_, _) | StatementKind::EndRegion(_) | StatementKind::UserAssertTy(_, _) => {}
        StatementKind::InlineAsm { .. } => fx.tcx.sess.fatal("Inline assembly is not supported"),
    }

    Ok(())
}

macro_rules! binop_match {
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $lhs:expr, $rhs:expr, bug) => {
        bug!("bin op {} on {} lhs: {:?} rhs: {:?}", stringify!($var), $bug_fmt, $lhs, $rhs)
    };
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $lhs:expr, $rhs:expr, icmp($cc:ident)) => {{
        let b = $fx.bcx.ins().icmp(IntCC::$cc, $lhs, $rhs);
        $fx.bcx.ins().bint(types::I8, b)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $lhs:expr, $rhs:expr, $name:ident) => {
        $fx.bcx.ins().$name($lhs, $rhs)
    };
    (
        $fx:expr, $bin_op:expr, $signed:expr, $lhs:expr, $rhs:expr, $bug_fmt:expr;
        $(
            $var:ident ($sign:pat) $name:tt $( ( $next:tt ) )? ;
        )*
    ) => {
        match ($bin_op, $signed) {
            $(
                (BinOp::$var, $sign) => binop_match!(@single $fx, $bug_fmt, $var, $lhs, $rhs, $name $( ( $next ) )?),
            )*
        }
    }
}

fn trans_int_binop<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, bin_op: BinOp, lhs: Value, rhs: Value, ty: Ty<'tcx>, signed: bool, _checked: bool) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, signed, lhs, rhs, "non ptr";
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
    };

    // TODO: return correct value for checked binops
    CValue::ByVal(res, fx.layout_of(ty))
}

fn trans_ptr_binop<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, bin_op: BinOp, lhs: Value, rhs: Value, ty: Ty<'tcx>, _checked: bool) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, "ptr";
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
    };

    // TODO: return correct value for checked binops
    CValue::ByVal(res, fx.layout_of(ty))
}

pub fn trans_place<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, place: &Place<'tcx>) -> CPlace<'tcx> {
    match place {
        Place::Local(local) => fx.get_local_place(*local),
        Place::Static(static_) => unimplemented!("static place {:?} ty {:?}", static_.def_id, static_.ty),
        Place::Projection(projection) => {
            let base = trans_place(fx, &projection.base);
            match projection.elem {
                ProjectionElem::Deref => {
                    CPlace::Addr(base.to_cvalue(fx).load_value(fx), fx.layout_of(place.ty(&*fx.mir, fx.tcx).to_ty(fx.tcx)))
                }
                ProjectionElem::Field(field, _ty) => {
                    base.place_field(fx, field)
                }
                ProjectionElem::Index(local) => unimplemented!("projection index {:?} {:?}", projection.base, local),
                ProjectionElem::ConstantIndex { offset, min_length: _, from_end: false } => unimplemented!("projection const index {:?} offset {:?} not from end", projection.base, offset),
                ProjectionElem::ConstantIndex { offset, min_length: _, from_end: true } => unimplemented!("projection const index {:?} offset {:?} from end", projection.base, offset),
                ProjectionElem::Subslice { from, to } => unimplemented!("projection subslice {:?} from {} to {}", projection.base, from, to),
                ProjectionElem::Downcast(_adt_def, variant) => {
                    base.downcast_variant(fx, variant)
                }
            }
        }
    }
}

pub fn trans_operand<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx>, operand: &Operand<'tcx>) -> CValue<'tcx> {
    match operand {
        Operand::Move(place) |
        Operand::Copy(place) => {
            let cplace = trans_place(fx, place);
            cplace.to_cvalue(fx)
        },
        Operand::Constant(const_) => {
            ::constant::trans_constant(fx, const_)
        }
    }
}
