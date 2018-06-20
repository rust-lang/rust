use syntax::ast::{IntTy, UintTy};
use rustc_mir::monomorphize::MonoItem;

use cretonne::prelude::*;
//use cretonne::codegen::Context;
use cretonne::codegen::ir::{
    ExternalName,
    FuncRef,
    StackSlot,
    function::Function,
};

use cretonne_module::{Module, Backend, FuncId, Linkage};
use cretonne_simplejit::{SimpleJITBuilder, SimpleJITBackend};

use std::any::Any;
use std::collections::HashMap;

use prelude::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Variable(Local);

type CurrentBackend = SimpleJITBackend;

impl EntityRef for Variable {
    fn new(u: usize) -> Self {
        Variable(Local::new(u))
    }

    fn index(self) -> usize {
        self.0.index()
    }
}

#[derive(Copy, Clone)]
enum CValue {
    ByRef(Value),
    ByVal(Value),
    Func(FuncRef),
}

impl CValue {
    fn force_stack<'a, 'tcx: 'a>(self, fx: &mut FunctionCx<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
        match self {
            CValue::ByRef(value) => value,
            CValue::ByVal(value) => {
                let layout = fx.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: layout.size.bytes() as u32,
                    offset: None,
                });
                fx.bcx.ins().stack_store(value, stack_slot, 0);
                fx.bcx.ins().stack_addr(types::I64, stack_slot, 0)
            }
            CValue::Func(func) => {
                let func = fx.bcx.ins().func_addr(types::I64, func);
                CValue::ByVal(func).force_stack(fx, ty)
            }
        }
    }

    fn load_value<'a, 'tcx: 'a>(self, fx: &mut FunctionCx<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
        match self {
            CValue::ByRef(value) => {
                let cton_ty = cton_type_from_ty(ty).unwrap();
                fx.bcx.ins().load(cton_ty, MemFlags::new(), value, 0)
            }
            CValue::ByVal(value) => value,
            CValue::Func(func) => {
                fx.bcx.ins().func_addr(types::I64, func)
            }
        }
    }

    fn expect_byref(self) -> Value {
        match self {
            CValue::ByRef(value) => value,
            CValue::ByVal(_) => unimplemented!("Expected CValue::ByRef, found CValue::ByVal"),
            CValue::Func(_) => unimplemented!("Expected CValue::ByRef, found CValue::Func"),
        }
    }
}

#[derive(Copy, Clone)]
enum CPlace {
    Var(Variable),
    Addr(Value),
}

impl CPlace {
    fn to_cvalue<'a, 'tcx: 'a>(self, fx: &mut FunctionCx<'a, 'tcx>) -> CValue {
        match self {
            CPlace::Var(var) => CValue::ByVal(fx.bcx.use_var(var)),
            CPlace::Addr(addr) => CValue::ByRef(addr),
        }
    }

    fn expect_addr(self) -> Value {
        match self {
            CPlace::Addr(addr) => addr,
            CPlace::Var(_) => unreachable!("Expected CPlace::Addr, found CPlace::Var"),
        }
    }
}

pub fn trans_crate<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Box<Any> {
    let link_meta = ::build_link_meta(tcx.crate_hash(LOCAL_CRATE));
    let metadata = tcx.encode_metadata(&link_meta);

    let module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new());
    //let mut context = Context::new();

    let mut cx = CodegenCx {
        tcx,
        module,
        def_id_fn_id_map: HashMap::new(),
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

                    let mut f = Function::with_name_signature(ext_name_from_did(def_id), sig);

                    trans_fn(cx, &mut f, def_id, substs);

                    let mut mir = ::std::io::Cursor::new(Vec::new());
                    ::rustc_mir::util::write_mir_pretty(cx.tcx, Some(def_id), &mut mir).unwrap();
                    let mut cton = String::new();
                    ::cretonne::codegen::write_function(&mut cton, &f, None).unwrap();
                    tcx.sess.warn(&format!("{:?}:\n\n{}\n\n{}", def_id, String::from_utf8_lossy(&mir.into_inner()), cton));

                    let flags = settings::Flags::new(settings::builder());
                    match ::cretonne::codegen::verify_function(&f, &flags) {
                        Ok(_) => {}
                        Err(err) => {
                            tcx.sess.fatal(&format!("cretonne verify error: {}", err));
                        }
                    }

                    //context.func = f;
                    //cx.module.define_function(func_id, &mut context).unwrap();
                    //context.clear();
                }
                _ => {}
            }
            _ => {}
        }
    }

    //cx.module.finalize_all();
    //cx.module.finish();

    tcx.sess.fatal("unimplemented");

    Box::new(::OngoingCodegen {
        metadata: metadata,
        //translated_module: Module::new(::cretonne_faerie::FaerieBuilder::new(,
        crate_name: tcx.crate_name(LOCAL_CRATE),
    })
}

struct CodegenCx<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: Module<CurrentBackend>,
    def_id_fn_id_map: HashMap<Instance<'tcx>, FuncId>,
}

struct FunctionCx<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &'a mut Module<CurrentBackend>,
    def_id_fn_id_map: &'a mut HashMap<Instance<'tcx>, FuncId>,
    bcx: FunctionBuilder<'a, Variable>,
    mir: &'tcx Mir<'tcx>,
    ebb_map: HashMap<BasicBlock, Ebb>,
    local_map: HashMap<Local, CPlace>,
}

impl<'f, 'tcx> FunctionCx<'f, 'tcx> {
    fn get_ebb(&self, bb: BasicBlock) -> Ebb {
        *self.ebb_map.get(&bb).unwrap()
    }

    fn get_local_place(&mut self, local: Local) -> CPlace {
        *self.local_map.get(&local).unwrap()
    }

    fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let tcx = self.tcx;
        let module = &mut self.module;
        let func_id = *self.def_id_fn_id_map.entry(inst).or_insert_with(|| {
            let sig = cton_sig_from_instance(tcx, inst);
            module.declare_function(&tcx.absolute_item_path_str(inst.def_id()), Linkage::Local, &sig).unwrap()
        });
        module.declare_func_in_func(func_id, &mut self.bcx.func)
    }
}

fn trans_fn<'a, 'tcx: 'a>(cx: &mut CodegenCx<'a, 'tcx>, f: &mut Function, def_id: DefId, substs: &Substs<'tcx>) {
    let mir = cx.tcx.optimized_mir(def_id);
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
        bcx,
        mir,
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
        let layout = fx.tcx.layout_of(ParamEnv::reveal_all().and(mir.local_decls[local].ty)).unwrap();
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let ty = cton_type_from_ty(mir.local_decls[local].ty);
        (local, fx.bcx.append_ebb_param(start_ebb, ty.unwrap_or(types::I64)), ty, stack_slot)
    }).collect::<Vec<(Local, Value, Option<Type>, StackSlot)>>();

    fx.local_map.insert(RETURN_PLACE, CPlace::Addr(ret_param));

    for (local, ebb_param, ty, stack_slot) in func_params {
        let addr = fx.bcx.ins().stack_addr(types::I64, stack_slot, 0);
        if ty.is_some() {
            fx.bcx.ins().stack_store(ebb_param, stack_slot, 0);
        } else {
            do_memcpy(fx, CPlace::Addr(addr), CValue::ByRef(ebb_param), mir.local_decls[local].ty);
        }
        fx.local_map.insert(local, CPlace::Addr(addr));
    }

    for local in mir.vars_and_temps_iter() {
        let layout = cx.tcx.layout_of(ParamEnv::reveal_all().and(mir.local_decls[local].ty)).unwrap();
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let addr = fx.bcx.ins().stack_addr(types::I64, stack_slot, 0);
        fx.local_map.insert(local, CPlace::Addr(addr));
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
                let cond_ty = cond.ty(&fx.mir.local_decls, fx.tcx);
                let cond = trans_operand(fx, cond).load_value(fx, cond_ty);
                let target = fx.get_ebb(*target);
                if *expected {
                    fx.bcx.ins().brz(cond, target, &[]);
                } else {
                    fx.bcx.ins().brnz(cond, target, &[]);
                }
                fx.bcx.ins().trap(TrapCode::User(!0));
            }

            TerminatorKind::SwitchInt { discr, switch_ty, values, targets } => {
                let discr_ty = discr.ty(&fx.mir.local_decls, fx.tcx);
                let discr = trans_operand(fx, discr).load_value(fx, discr_ty);
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
                                if let Some(_) = cton_type_from_ty(ty) {
                                    arg.load_value(fx, ty)
                                } else {
                                    arg.force_stack(fx, ty)
                                }
                            })
                    ).collect::<Vec<_>>();
                match func {
                    CValue::Func(func) => {
                        fx.bcx.ins().call(func, &args);
                    }
                    func => {
                        let func = func.load_value(fx, func_ty);
                        let sig = match func_ty.sty {
                            TypeVariants::TyFnDef(def_id, _substs) => fx.tcx.fn_sig(def_id),
                            TypeVariants::TyFnPtr(fn_sig) => fn_sig,
                            _ => bug!("Calling non function type {:?}", func_ty),
                        };
                        let sig = fx.bcx.import_signature(cton_sig_from_fn_sig(fx.tcx, sig, substs));
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
        StatementKind::Assign(place, rval) => {
            let ty = place.ty(&fx.mir.local_decls, fx.tcx).to_ty(fx.tcx);
            let lval = trans_place(fx, place);
            let rval = trans_rval(fx, rval);
            do_memcpy(fx, lval, CValue::ByRef(rval), ty);
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {}
        _ => unimplemented!("stmt {:?}", stmt),
    }
}

fn trans_place<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, place: &Place<'tcx>) -> CPlace {
    match place {
        Place::Local(local) => fx.get_local_place(*local),
        Place::Projection(projection) => {
            let base = trans_place(fx, &projection.base).expect_addr();
            match projection.elem {
                ProjectionElem::Field(field, ty) => {
                    let layout = fx.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                    let field_offset = layout.fields.offset(field.index());
                    if field_offset.bytes() > 0 {
                        let field_offset = fx.bcx.ins().iconst(types::I64, field_offset.bytes() as i64);
                        CPlace::Addr(fx.bcx.ins().iadd(base, field_offset))
                    } else {
                        CPlace::Addr(base)
                    }
                }
                _ => unimplemented!("projection {:?}", projection),
            }
        }
        place => unimplemented!("place {:?}", place),
    }
}

fn trans_rval<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, rval: &Rvalue<'tcx>) -> Value {
    match rval {
        Rvalue::Use(operand) => {
            let operand_ty = operand.ty(&fx.mir.local_decls, fx.tcx);
            trans_operand(fx, operand).force_stack(fx, operand_ty)
        },
        Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
            match bin_op {
                BinOp::Mul => {
                    let ty = lhs.ty(&fx.mir.local_decls, fx.tcx);
                    let lhs_ty = lhs.ty(&fx.mir.local_decls, fx.tcx);
                    let lhs = trans_operand(fx, lhs).load_value(fx, lhs_ty);
                    let rhs_ty = rhs.ty(&fx.mir.local_decls, fx.tcx);
                    let rhs = trans_operand(fx, rhs).load_value(fx, rhs_ty);
                    let res = match ty.sty {
                        TypeVariants::TyUint(_) => {
                            fx.bcx.ins().imul(lhs, rhs)
                        }
                        _ => unimplemented!(),
                    };
                    let layout = fx.tcx.layout_of(ParamEnv::empty().and(rval.ty(&fx.mir.local_decls, fx.tcx))).unwrap();
                    let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: layout.size.bytes() as u32,
                        offset: None,
                    });
                    fx.bcx.ins().stack_store(res, stack_slot, 1);
                    fx.bcx.ins().stack_addr(types::I64, stack_slot, 1)
                }
                bin_op => unimplemented!("checked bin op {:?} {:?} {:?}", bin_op, lhs, rhs),
            }
        }
        Rvalue::Cast(CastKind::ReifyFnPointer, operand, ty) => {
            let operand = trans_operand(fx, operand);
            operand.force_stack(fx, ty)
        }
        Rvalue::Cast(CastKind::UnsafeFnPointer, operand, ty) => {
            trans_operand(fx, operand).force_stack(fx, ty)
        }
        rval => unimplemented!("rval {:?}", rval),
    }
}

fn trans_operand<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx>, operand: &Operand<'tcx>) -> CValue {
    match operand {
        Operand::Move(place) |
        Operand::Copy(place) => {
            let cplace = trans_place(fx, place);
            cplace.to_cvalue(fx)
        },
        Operand::Constant(const_) => {
            match const_.literal {
                Literal::Value { value } => {
                    let layout = fx.tcx.layout_of(ParamEnv::empty().and(const_.ty)).unwrap();
                    match const_.ty.sty {
                        TypeVariants::TyUint(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            let iconst = fx.bcx.ins().iconst(cton_type_from_ty(const_.ty).unwrap(), bits as u64 as i64);
                            CValue::ByVal(iconst)
                        }
                        TypeVariants::TyInt(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            let iconst = fx.bcx.ins().iconst(cton_type_from_ty(const_.ty).unwrap(), bits as i128 as i64);
                            CValue::ByVal(iconst)
                        }
                        TypeVariants::TyFnDef(def_id, substs) => {
                            let func_ref = fx.get_function_ref(Instance::new(def_id, substs));
                            CValue::Func(func_ref)
                        }
                        _ => unimplemented!("value {:?} ty {:?}", value, const_.ty),
                    }
                }
                _ => unimplemented!()
            }
        }
    }
}

fn do_memcpy<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, to: CPlace, from: CValue, ty: Ty<'tcx>) {
    let layout = fx.tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();
    let size = layout.size.bytes() as i32;
    match to {
        CPlace::Var(var) => {
            let data = from.load_value(fx, ty);
            fx.bcx.def_var(var, data)
        },
        CPlace::Addr(addr) => {
            if cton_type_from_ty(ty).is_some() {
                let data = from.load_value(fx, ty);
                fx.bcx.ins().store(MemFlags::new(), data, addr, 0);
            } else {
                for i in 0..size {
                    let from = from.expect_byref();
                    let byte = fx.bcx.ins().load(types::I8, MemFlags::new(), from, i);
                    fx.bcx.ins().store(MemFlags::new(), byte, addr, i);
                }
            }
        }
    }
}

fn ext_name_from_did(def_id: DefId) -> ExternalName {
    ExternalName::user(def_id.krate.as_u32(), def_id.index.as_raw_u32())
}

fn cton_sig_from_fn_sig<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sig: PolyFnSig<'tcx>, substs: &Substs<'tcx>) -> Signature {
    let sig = tcx.subst_and_normalize_erasing_regions(substs, ParamEnv::reveal_all(), &sig);
    cton_sig_from_mono_fn_sig(sig)
}

fn cton_sig_from_instance<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, inst: Instance<'tcx>) -> Signature {
    let fn_ty = inst.ty(tcx);
    let sig = fn_ty.fn_sig(tcx);
    cton_sig_from_mono_fn_sig(sig)
}

fn cton_sig_from_mono_fn_sig<'a ,'tcx: 'a>(sig: PolyFnSig<'tcx>) -> Signature {
    let sig = sig.skip_binder();
    let inputs = sig.inputs();
    let _output = sig.output();
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let call_conv = match sig.abi {
        _ => CallConv::SystemV,
    };
    Signature {
        params: Some(types::I64).into_iter() // First param is place to put return val
            .chain(inputs.into_iter().map(|ty| cton_type_from_ty(ty).unwrap_or(types::I64)))
            .map(AbiParam::new).collect(),
        returns: vec![],
        call_conv,
        argument_bytes: None,
    }
}

fn cton_type_from_ty(ty: Ty) -> Option<types::Type> {
    Some(match ty.sty {
        TypeVariants::TyBool => types::I8,
        TypeVariants::TyUint(size) => {
            match size {
                UintTy::U8 => types::I8,
                UintTy::U16 => types::I16,
                UintTy::U32 => types::I32,
                UintTy::U64 => types::I64,
                UintTy::U128 => unimplemented!(),
                UintTy::Usize => unimplemented!(),
            }
        }
        TypeVariants::TyInt(size) => {
            match size {
                IntTy::I8 => types::I8,
                IntTy::I16 => types::I16,
                IntTy::I32 => types::I32,
                IntTy::I64 => types::I64,
                IntTy::I128 => unimplemented!(),
                IntTy::Isize => unimplemented!(),
            }
        }
        TypeVariants::TyFnPtr(_) => types::I64,
        _ => return None,
    })
}
