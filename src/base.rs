use syntax::ast::{IntTy, UintTy};
use rustc_mir::monomorphize::MonoItem;

use cretonne::prelude::*;
use cretonne::codegen::ir::{
    ExternalName,
    FuncRef,
    function::Function,
};

use std::any::Any;
use std::collections::HashMap;

use prelude::*;

pub struct Translated {
    f: Function,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Variable(Local);

impl EntityRef for Variable {
    fn new(u: usize) -> Self {
        Variable(Local::new(u))
    }

    fn index(self) -> usize {
        self.0.index()
    }
}

enum CValue {
    ByRef(Value),
    ByVal(Value),
    Func(FuncRef),
}

impl CValue {
    fn force_stack<'a, 'tcx: 'a>(self, ccx: &mut CodegenCtxt<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
        match self {
            CValue::ByRef(value) => value,
            CValue::ByVal(value) => {
                let layout = ccx.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                let stack_slot = ccx.bcx.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: layout.size.bytes() as u32,
                    offset: None,
                });
                ccx.bcx.ins().stack_store(value, stack_slot, 0);
                ccx.bcx.ins().stack_addr(types::I64, stack_slot, 0)
            }
            CValue::Func(func) => {
                let func = ccx.bcx.ins().func_addr(types::I64, func);
                CValue::ByVal(func).force_stack(ccx, ty)
            }
        }
    }

    fn load_value<'a, 'tcx: 'a>(self, ccx: &mut CodegenCtxt<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
        match self {
            CValue::ByRef(value) => {
                let cton_ty = cton_type_from_ty(ty).unwrap();
                ccx.bcx.ins().load(cton_ty, MemFlags::new(), value, 0)
            }
            CValue::ByVal(value) => value,
            CValue::Func(func) => {
                ccx.bcx.ins().func_addr(types::I64, func)
            }
        }
    }
}

pub fn trans_crate<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Box<Any> {
    let link_meta = ::build_link_meta(tcx.crate_hash(LOCAL_CRATE));
    let metadata = tcx.encode_metadata(&link_meta);

    let mut translated_mono_items = Vec::new();

    for mono_item in
        collector::collect_crate_mono_items(
            tcx,
            collector::MonoItemCollectionMode::Eager
        ).0 {
        match mono_item {
            MonoItem::Fn(Instance {
                def: InstanceDef::Item(def_id),
                substs,
            }) => {
                let sig = tcx.fn_sig(def_id);
                let sig = tcx.subst_and_normalize_erasing_regions(substs, ParamEnv::reveal_all(), &sig);
                let mut f = Function::with_name_signature(ext_name_from_did(def_id), cton_sig_from_fn_sig(sig.skip_binder()));

                trans_fn(tcx, &mut f, def_id, substs);

                let flags = settings::Flags::new(settings::builder());
                let verify_error: String = ::cretonne::codegen::verify_function(&f, &flags)
                    .map(|_| String::new())
                    .unwrap_or_else(|err| format!("\n\ncretonne error: {}", err));

                let mut mir = ::std::io::Cursor::new(Vec::new());
                ::rustc_mir::util::write_mir_pretty(tcx, Some(def_id), &mut mir).unwrap();
                let mut cton = String::new();
                ::cretonne::codegen::write_function(&mut cton, &f, None).unwrap();
                tcx.sess.warn(&format!("{:?}:\n\n{}\n\n{}{}", def_id, String::from_utf8_lossy(&mir.into_inner()), cton, verify_error));

                translated_mono_items.push(Translated {
                    f,
                });
            }
            _ => {}
        }
    }

    Box::new(::OngoingCodegen {
        metadata: metadata,
        translated_mono_items,
        crate_name: tcx.crate_name(LOCAL_CRATE),
    })
}

struct CodegenCtxt<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    bcx: FunctionBuilder<'a, Variable>,
    mir: &'tcx Mir<'tcx>,
    ebb_map: HashMap<BasicBlock, Ebb>,
    args_map: HashMap<Local, Value>,
}

impl<'f, 'tcx> CodegenCtxt<'f, 'tcx> {
    fn get_ebb(&self, bb: BasicBlock) -> Ebb {
        *self.ebb_map.get(&bb).unwrap()
    }

    fn get_local(&mut self, local: Local) -> Value {
        match self.mir.local_kind(local) {
            LocalKind::Arg => *self.args_map.get(&local).unwrap(),
            LocalKind::ReturnPointer => *self.args_map.get(&RETURN_PLACE).unwrap(),
            LocalKind::Temp | LocalKind::Var => self.bcx.use_var(Variable(local)),
        }
    }
}

fn trans_fn<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, f: &mut Function, def_id: DefId, substs: &Substs) {
    let mir = tcx.optimized_mir(def_id);
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx: FunctionBuilder<Variable> = FunctionBuilder::new(f, &mut func_ctx);

    let start_ebb = bcx.create_ebb();
    bcx.switch_to_block(start_ebb);
    let mut ebb_map: HashMap<BasicBlock, Ebb> = HashMap::new();
    for (bb, _bb_data) in mir.basic_blocks().iter_enumerated() {
        ebb_map.insert(bb, bcx.create_ebb());
    }

    let mut args_map: HashMap<Local, Value> = HashMap::new();
    for arg in Some(RETURN_PLACE).into_iter().chain(mir.args_iter()) {
        let ty = types::I64;
        args_map.insert(arg, bcx.append_ebb_param(start_ebb, ty));
    }

    for local in mir.vars_and_temps_iter() {
        let layout = tcx.layout_of(ParamEnv::reveal_all().and(mir.local_decls[local].ty)).unwrap();
        let stack_slot = bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let ty = types::I64;
        bcx.declare_var(Variable(local), ty);
        let val = bcx.ins().stack_addr(ty, stack_slot, 0);
        bcx.def_var(Variable(local), val);
    }
    bcx.ins().jump(*ebb_map.get(&START_BLOCK).unwrap(), &[]);

    let mut ccx = CodegenCtxt {
        tcx,
        bcx,
        mir,
        ebb_map,
        args_map,
    };
    let ccx = &mut ccx;

    for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
        let ebb = ccx.get_ebb(bb);
        ccx.bcx.switch_to_block(ebb);

        for stmt in &bb_data.statements {
            trans_stmt(ccx, stmt);
        }

        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                let ebb = ccx.get_ebb(*target);
                ccx.bcx.ins().jump(ebb, &[]);
            }
            TerminatorKind::Return => {
                ccx.bcx.ins().return_(&[]);
            }
            TerminatorKind::Assert { cond, expected, msg, target, cleanup: _ } => {
                let cond_ty = cond.ty(&ccx.mir.local_decls, ccx.tcx);
                let cond = trans_operand(ccx, cond).load_value(ccx, cond_ty);
                let target = ccx.get_ebb(*target);
                if *expected {
                    ccx.bcx.ins().brz(cond, target, &[]);
                } else {
                    ccx.bcx.ins().brnz(cond, target, &[]);
                }
                ccx.bcx.ins().trap(TrapCode::User(!0));
            }

            TerminatorKind::SwitchInt { discr, switch_ty, values, targets } => {
                let discr_ty = discr.ty(&ccx.mir.local_decls, ccx.tcx);
                let discr = trans_operand(ccx, discr).load_value(ccx, discr_ty);
                let mut jt_data = JumpTableData::new();
                for (i, value) in values.iter().enumerate() {
                    let ebb = ccx.get_ebb(targets[i]);
                    jt_data.set_entry(*value as usize, ebb);
                }
                let mut jump_table = ccx.bcx.create_jump_table(jt_data);
                ccx.bcx.ins().br_table(discr, jump_table);
                let otherwise_ebb = ccx.get_ebb(targets[targets.len() - 1]);
                ccx.bcx.ins().jump(otherwise_ebb, &[]);
            }
            TerminatorKind::Call { func, args, destination, cleanup: _ } => {
                let func = trans_operand(ccx, func);
                let return_place = if let Some((place, _)) = destination {
                    trans_place(ccx, place)
                } else {
                    ccx.bcx.ins().iconst(types::I64, 0)
                };
                let args = Some(return_place)
                    .into_iter()
                    .chain(
                        args
                            .into_iter()
                            .map(|arg| {
                                let ty = arg.ty(&ccx.mir.local_decls, ccx.tcx);
                                let arg = trans_operand(ccx, arg);
                                arg.force_stack(ccx, ty)
                            })
                    ).collect::<Vec<_>>();
                match func {
                    CValue::Func(func) => {
                        ccx.bcx.ins().call(func, &args);
                    }
                    _ => unimplemented!("indirect call"),
                }
                if let Some((_, dest)) = *destination {
                    let ret_ebb = ccx.get_ebb(dest);
                    ccx.bcx.ins().jump(ret_ebb, &[]);
                } else {
                    ccx.bcx.ins().trap(TrapCode::User(!0));
                }
            }
            TerminatorKind::Resume | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                ccx.bcx.ins().trap(TrapCode::User(!0));
            }
            terminator => unimplemented!("terminator {:?}", terminator),
        }
    }

    ccx.bcx.seal_all_blocks();
    ccx.bcx.finalize();
}

fn trans_stmt<'a, 'tcx: 'a>(ccx: &mut CodegenCtxt<'a, 'tcx>, stmt: &Statement<'tcx>) {
    match &stmt.kind {
        StatementKind::Assign(place, rval) => {
            let ty = place.ty(&ccx.mir.local_decls, ccx.tcx).to_ty(ccx.tcx);
            let lval = trans_place(ccx, place);
            let rval = trans_rval(ccx, rval);
            do_memcpy(ccx, lval, rval, ty);
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {}
        _ => unimplemented!("stmt {:?}", stmt),
    }
}

fn trans_place<'a, 'tcx: 'a>(ccx: &mut CodegenCtxt<'a, 'tcx>, place: &Place<'tcx>) -> Value {
    match place {
        Place::Local(local) => ccx.get_local(*local),
        Place::Projection(projection) => {
            let base = trans_place(ccx, &projection.base);
            match projection.elem {
                ProjectionElem::Field(field, ty) => {
                    let layout = ccx.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                    let field_offset = layout.fields.offset(field.index());
                    let field_offset = ccx.bcx.ins().iconst(types::I64, field_offset.bytes() as i64);
                    ccx.bcx.ins().iadd(base, field_offset)
                }
                _ => unimplemented!("projection {:?}", projection),
            }
        }
        place => unimplemented!("place {:?}", place),
    }
}

fn trans_rval<'a, 'tcx: 'a>(ccx: &mut CodegenCtxt<'a, 'tcx>, rval: &Rvalue<'tcx>) -> Value {
    match rval {
        Rvalue::Use(operand) => {
            let operand_ty = operand.ty(&ccx.mir.local_decls, ccx.tcx);
            trans_operand(ccx, operand).force_stack(ccx, operand_ty)
        },
        Rvalue::CheckedBinaryOp(bin_op, lhs, rhs) => {
            match bin_op {
                BinOp::Mul => {
                    let ty = lhs.ty(&ccx.mir.local_decls, ccx.tcx);
                    let lhs_ty = lhs.ty(&ccx.mir.local_decls, ccx.tcx);
                    let lhs = trans_operand(ccx, lhs).load_value(ccx, lhs_ty);
                    let rhs_ty = rhs.ty(&ccx.mir.local_decls, ccx.tcx);
                    let rhs = trans_operand(ccx, rhs).load_value(ccx, rhs_ty);
                    match ty.sty {
                        TypeVariants::TyUint(_) => {
                            ccx.bcx.ins().imul(lhs, rhs)
                        }
                        _ => unimplemented!(),
                    }
                }
                bin_op => unimplemented!("checked bin op {:?} {:?} {:?}", bin_op, lhs, rhs),
            }
        }
        rval => unimplemented!("{:?}", rval),
    }
}

fn trans_operand<'a, 'tcx>(ccx: &mut CodegenCtxt<'a, 'tcx>, operand: &Operand<'tcx>) -> CValue {
    match operand {
        Operand::Move(place) => CValue::ByRef(trans_place(ccx, place)),
        Operand::Constant(const_) => {
            match const_.literal {
                Literal::Value { value } => {
                    let layout = ccx.tcx.layout_of(ParamEnv::empty().and(const_.ty)).unwrap();
                    match const_.ty.sty {
                        TypeVariants::TyUint(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            let iconst = ccx.bcx.ins().iconst(cton_type_from_ty(const_.ty).unwrap(), bits as u64 as i64);
                            CValue::ByVal(iconst)
                        }
                        TypeVariants::TyInt(_) => {
                            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
                            let iconst = ccx.bcx.ins().iconst(cton_type_from_ty(const_.ty).unwrap(), bits as i128 as i64);
                            CValue::ByVal(iconst)
                        }
                        TypeVariants::TyFnDef(def_id, substs) => {
                            let ext_name = ext_name_from_did(def_id);
                            let sig = ccx.tcx.fn_sig(def_id);
                            let sig = ccx.tcx.subst_and_normalize_erasing_regions(substs, ParamEnv::reveal_all(), &sig);
                            let sig = ccx.bcx.import_signature(cton_sig_from_fn_sig(sig.skip_binder()));
                            CValue::Func(ccx.bcx.import_function(ExtFuncData {
                                name: ext_name,
                                signature: sig,
                                colocated: false,
                            }))
                        }
                        _ => unimplemented!("value {:?} ty {:?}", value, const_.ty),
                    }
                }
                _ => unimplemented!()
            }
        }
        operand => unimplemented!("operand {:?}", operand),
    }
}

fn do_memcpy<'a, 'tcx: 'a>(ccx: &mut CodegenCtxt<'a, 'tcx>, to: Value, from: Value, ty: Ty<'tcx>) {
    let layout = ccx.tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();
    for i in 0..(layout.size.bytes() as i32) {
        let byte = ccx.bcx.ins().load_complex(types::I8, MemFlags::new(), &[from], i);
        ccx.bcx.ins().store_complex(MemFlags::new(), byte, &[to], i);
    }
}

fn ext_name_from_did(def_id: DefId) -> ExternalName {
    ExternalName::user(def_id.krate.as_u32(), def_id.index.as_raw_u32())
}

fn cton_sig_from_fn_sig(sig: &FnSig) -> Signature {
    let inputs = sig.inputs();
    let output = sig.output();
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let call_conv = match sig.abi {
        _ => CallConv::SystemV,
    };
    Signature {
        params: Some(types::I64).into_iter() // First param is palce to put return val
            .chain(inputs.into_iter().map(|_| types::I64))
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
        _ => return None,
    })
}
