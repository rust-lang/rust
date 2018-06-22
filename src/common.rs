use syntax::ast::{IntTy, UintTy};

use cretonne_module::{Module, Linkage, FuncId};

use prelude::*;

pub type CurrentBackend = ::cretonne_simplejit::SimpleJITBackend;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Variable(Local);

impl EntityRef for Variable {
    fn new(u: usize) -> Self {
        Variable(Local::new(u))
    }

    fn index(self) -> usize {
        self.0.index()
    }
}

pub fn cton_type_from_ty(ty: Ty) -> Option<types::Type> {
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

// FIXME(cretonne) fix load.i8
fn load_workaround(fx: &mut FunctionCx, ty: Type, addr: Value, offset: i32) -> Value {
    use cretonne::codegen::ir::types::*;
    match ty {
        I8 => fx.bcx.ins().uload8(I32, MemFlags::new(), addr, offset),
        I16 => fx.bcx.ins().uload16(I32, MemFlags::new(), addr, offset),
        // I32 and I64 work
        _ => fx.bcx.ins().load(ty, MemFlags::new(), addr, offset),
    }
}

// FIXME(cretonne) fix store.i8
fn store_workaround(fx: &mut FunctionCx, ty: Type, addr: Value, val: Value, offset: i32) {
    use cretonne::codegen::ir::types::*;
    match ty {
        I8 => fx.bcx.ins().istore8(MemFlags::new(), val, addr, offset),
        I16 => fx.bcx.ins().istore16(MemFlags::new(), val, addr, offset),
        // I32 and I64 work
        _ => fx.bcx.ins().store(MemFlags::new(), val, addr, offset),
    };
}

#[derive(Copy, Clone)]
pub enum CValue {
    ByRef(Value),
    ByVal(Value),
    Func(FuncRef),
}

impl CValue {
    pub fn force_stack<'a, 'tcx: 'a>(self, fx: &mut FunctionCx<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
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

    pub fn load_value<'a, 'tcx: 'a>(self, fx: &mut FunctionCx<'a, 'tcx>, ty: Ty<'tcx>) -> Value {
        match self {
            CValue::ByRef(value) => {
                let cton_ty = cton_type_from_ty(ty).unwrap();
                load_workaround(fx, cton_ty, value, 0)
            }
            CValue::ByVal(value) => value,
            CValue::Func(func) => {
                fx.bcx.ins().func_addr(types::I64, func)
            }
        }
    }

    pub fn expect_byref(self) -> Value {
        match self {
            CValue::ByRef(value) => value,
            CValue::ByVal(_) => bug!("Expected CValue::ByRef, found CValue::ByVal"),
            CValue::Func(_) => bug!("Expected CValue::ByRef, found CValue::Func"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum CPlace {
    Var(Variable),
    Addr(Value),
}

impl<'a, 'tcx: 'a> CPlace {
    pub fn from_stack_slot(fx: &mut FunctionCx<'a, 'tcx>, stack_slot: StackSlot) -> CPlace {
        CPlace::Addr(fx.bcx.ins().stack_addr(types::I64, stack_slot, 0))
    }

    pub fn to_cvalue(self, fx: &mut FunctionCx<'a, 'tcx>) -> CValue {
        match self {
            CPlace::Var(var) => CValue::ByVal(fx.bcx.use_var(var)),
            CPlace::Addr(addr) => CValue::ByRef(addr),
        }
    }

    pub fn expect_addr(self) -> Value {
        match self {
            CPlace::Addr(addr) => addr,
            CPlace::Var(_) => bug!("Expected CPlace::Addr, found CPlace::Var"),
        }
    }

    pub fn write_cvalue(self, fx: &mut FunctionCx<'a, 'tcx>, from: CValue, ty: Ty<'tcx>) {
        let layout = fx.tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();
        let size = layout.size.bytes() as i32;
        match self {
            CPlace::Var(var) => {
                let data = from.load_value(fx, ty);
                fx.bcx.def_var(var, data)
            },
            CPlace::Addr(addr) => {
                if let Some(cton_ty) = cton_type_from_ty(ty) {
                    let data = from.load_value(fx, ty);
                    store_workaround(fx, cton_ty, addr, data, 0);
                } else {
                    for i in 0..size {
                        let from = from.expect_byref();
                        let byte = load_workaround(fx, types::I8, from, i);
                        store_workaround(fx, types::I8, addr, byte, i);
                    }
                }
            }
        }
    }
}

pub fn ext_name_from_did(def_id: DefId) -> ExternalName {
    ExternalName::user(def_id.krate.as_u32(), def_id.index.as_raw_u32())
}

pub fn cton_sig_from_fn_sig<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sig: PolyFnSig<'tcx>, substs: &Substs<'tcx>) -> Signature {
    let sig = tcx.subst_and_normalize_erasing_regions(substs, ParamEnv::reveal_all(), &sig);
    cton_sig_from_mono_fn_sig(sig)
}

pub fn cton_sig_from_instance<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, inst: Instance<'tcx>) -> Signature {
    let fn_ty = inst.ty(tcx);
    let sig = fn_ty.fn_sig(tcx);
    cton_sig_from_mono_fn_sig(sig)
}

pub fn cton_sig_from_mono_fn_sig<'a ,'tcx: 'a>(sig: PolyFnSig<'tcx>) -> Signature {
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

pub struct FunctionCx<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub module: &'a mut Module<CurrentBackend>,
    pub def_id_fn_id_map: &'a mut HashMap<Instance<'tcx>, FuncId>,
    pub bcx: FunctionBuilder<'a, Variable>,
    pub mir: &'tcx Mir<'tcx>,
    pub ebb_map: HashMap<BasicBlock, Ebb>,
    pub local_map: HashMap<Local, CPlace>,
}

impl<'f, 'tcx> FunctionCx<'f, 'tcx> {
    pub fn get_ebb(&self, bb: BasicBlock) -> Ebb {
        *self.ebb_map.get(&bb).unwrap()
    }

    pub fn get_local_place(&mut self, local: Local) -> CPlace {
        *self.local_map.get(&local).unwrap()
    }

    pub fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let tcx = self.tcx;
        let module = &mut self.module;
        let func_id = *self.def_id_fn_id_map.entry(inst).or_insert_with(|| {
            let sig = cton_sig_from_instance(tcx, inst);
            module.declare_function(&tcx.absolute_item_path_str(inst.def_id()), Linkage::Local, &sig).unwrap()
        });
        module.declare_func_in_func(func_id, &mut self.bcx.func)
    }
}
