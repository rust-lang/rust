use rustc_target::spec::{HasTargetSpec, Target};

use cranelift::codegen::ir::{Opcode, InstructionData, ValueDef};
use cranelift_module::Module;

use crate::prelude::*;

pub fn mir_var(loc: Local) -> Variable {
    Variable::with_u32(loc.index() as u32)
}

pub fn pointer_ty(tcx: TyCtxt) -> types::Type {
    match tcx.data_layout.pointer_size.bits() {
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        bits => bug!("ptr_sized_integer: unknown pointer bit size {}", bits),
    }
}

pub fn clif_type_from_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<types::Type> {
    Some(match ty.sty {
        ty::Bool => types::I8,
        ty::Uint(size) => match size {
            UintTy::U8 => types::I8,
            UintTy::U16 => types::I16,
            UintTy::U32 => types::I32,
            UintTy::U64 => types::I64,
            UintTy::U128 => types::I128,
            UintTy::Usize => pointer_ty(tcx),
        },
        ty::Int(size) => match size {
            IntTy::I8 => types::I8,
            IntTy::I16 => types::I16,
            IntTy::I32 => types::I32,
            IntTy::I64 => types::I64,
            IntTy::I128 => types::I128,
            IntTy::Isize => pointer_ty(tcx),
        },
        ty::Char => types::I32,
        ty::Float(size) => match size {
            FloatTy::F32 => types::F32,
            FloatTy::F64 => types::F64,
        },
        ty::FnPtr(_) => pointer_ty(tcx),
        ty::RawPtr(TypeAndMut { ty, mutbl: _ }) | ty::Ref(_, ty, _) => {
            if ty.is_sized(tcx.at(DUMMY_SP), ParamEnv::reveal_all()) {
                pointer_ty(tcx)
            } else {
                return None;
            }
        }
        ty::Param(_) => bug!("ty param {:?}", ty),
        _ => return None,
    })
}

pub fn codegen_select(bcx: &mut FunctionBuilder, cond: Value, lhs: Value, rhs: Value) -> Value {
    let lhs_ty = bcx.func.dfg.value_type(lhs);
    let rhs_ty = bcx.func.dfg.value_type(rhs);
    assert_eq!(lhs_ty, rhs_ty);
    if lhs_ty == types::I8 || lhs_ty == types::I16 {
        // FIXME workaround for missing encoding for select.i8
        let lhs = bcx.ins().uextend(types::I32, lhs);
        let rhs = bcx.ins().uextend(types::I32, rhs);
        let res = bcx.ins().select(cond, lhs, rhs);
        bcx.ins().ireduce(lhs_ty, res)
    } else {
        bcx.ins().select(cond, lhs, rhs)
    }
}

pub fn codegen_icmp<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    intcc: IntCC,
    lhs: Value,
    rhs: Value,
) -> Value {
    let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
    let rhs_ty = fx.bcx.func.dfg.value_type(rhs);
    assert_eq!(lhs_ty, rhs_ty);
    if lhs_ty == types::I128 {
        // FIXME legalize `icmp.i128` in Cranelift

        let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs);
        let (rhs_lsb, rhs_msb) = fx.bcx.ins().isplit(rhs);

        match intcc {
            IntCC::Equal => {
                let lsb_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_lsb, rhs_lsb);
                let msb_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_msb, rhs_msb);
                fx.bcx.ins().band(lsb_eq, msb_eq)
            }
            IntCC::NotEqual => {
                let lsb_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_lsb, rhs_lsb);
                let msb_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_msb, rhs_msb);
                fx.bcx.ins().bor(lsb_ne, msb_ne)
            }
            _ => {
                // if msb_eq {
                //     lsb_cc
                // } else {
                //     msb_cc
                // }

                let msb_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_msb, rhs_msb);
                let lsb_cc = fx.bcx.ins().icmp(intcc, lhs_lsb, rhs_lsb);
                let msb_cc = fx.bcx.ins().icmp(intcc, lhs_msb, rhs_msb);

                fx.bcx.ins().select(msb_eq, lsb_cc, msb_cc)
            }
        }
    } else {
        fx.bcx.ins().icmp(intcc, lhs, rhs)
    }
}

pub fn codegen_icmp_imm<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    intcc: IntCC,
    lhs: Value,
    rhs: i128,
) -> Value {
    let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
    if lhs_ty == types::I128 {
        // FIXME legalize `icmp_imm.i128` in Cranelift

        let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs);
        let (rhs_lsb, rhs_msb) = (rhs as u128 as u64 as i64, (rhs as u128 >> 64) as u64 as i64);

        match intcc {
            IntCC::Equal => {
                let lsb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_lsb, rhs_lsb);
                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                fx.bcx.ins().band(lsb_eq, msb_eq)
            }
            IntCC::NotEqual => {
                let lsb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_lsb, rhs_lsb);
                let msb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_msb, rhs_msb);
                fx.bcx.ins().bor(lsb_ne, msb_ne)
            }
            _ => {
                // if msb_eq {
                //     lsb_cc
                // } else {
                //     msb_cc
                // }

                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                let lsb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_lsb, rhs_lsb);
                let msb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_msb, rhs_msb);

                fx.bcx.ins().select(msb_eq, lsb_cc, msb_cc)
            }
        }
    } else {
        let rhs = i64::try_from(rhs).expect("codegen_icmp_imm rhs out of range for <128bit int");
        fx.bcx.ins().icmp_imm(intcc, lhs, rhs)
    }
}

fn resolve_normal_value_imm(func: &Function, val: Value) -> Option<i64> {
    if let ValueDef::Result(inst, 0 /*param*/) = func.dfg.value_def(val) {
        if let InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } = func.dfg[inst] {
            Some(imm.into())
        } else {
            None
        }
    } else {
        None
    }
}

fn resolve_128bit_value_imm(func: &Function, val: Value) -> Option<u128> {
    let (lsb, msb) = if let ValueDef::Result(inst, 0 /*param*/) = func.dfg.value_def(val) {
        if let InstructionData::Binary {
            opcode: Opcode::Iconcat,
            args: [lsb, msb],
        } = func.dfg[inst] {
            (lsb, msb)
        } else {
            return None;
        }
    } else {
        return None;
    };

    let lsb = resolve_normal_value_imm(func, lsb)? as u64 as u128;
    let msb = resolve_normal_value_imm(func, msb)? as u64 as u128;

    Some(msb << 64 | lsb)
}

pub fn resolve_value_imm(func: &Function, val: Value) -> Option<u128> {
    if func.dfg.value_type(val) == types::I128 {
        resolve_128bit_value_imm(func, val)
    } else {
        resolve_normal_value_imm(func, val).map(|imm| imm as u64 as u128)
    }
}

pub fn type_min_max_value(ty: Type, signed: bool) -> (i64, i64) {
    assert!(ty.is_int());
    let min = match (ty, signed) {
        (types::I8 , false)
        | (types::I16, false)
        | (types::I32, false)
        | (types::I64, false) => 0i64,
        (types::I8, true) => i8::min_value() as i64,
        (types::I16, true) => i16::min_value() as i64,
        (types::I32, true) => i32::min_value() as i64,
        (types::I64, true) => i64::min_value(),
        (types::I128, _) => unimplemented!(),
        _ => unreachable!(),
    };

    let max = match (ty, signed) {
        (types::I8, false) => u8::max_value() as i64,
        (types::I16, false) => u16::max_value() as i64,
        (types::I32, false) => u32::max_value() as i64,
        (types::I64, false) => u64::max_value() as i64,
        (types::I8, true) => i8::max_value() as i64,
        (types::I16, true) => i16::max_value() as i64,
        (types::I32, true) => i32::max_value() as i64,
        (types::I64, true) => i64::max_value(),
        (types::I128, _) => unimplemented!(),
        _ => unreachable!(),
    };

    (min, max)
}

pub fn type_sign(ty: Ty<'_>) -> bool {
    match ty.sty {
        ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::Char | ty::Uint(..) | ty::Bool => false,
        ty::Int(..) => true,
        ty::Float(..) => false, // `signed` is unused for floats
        _ => panic!("{}", ty),
    }
}

pub struct FunctionCx<'a, 'tcx: 'a, B: Backend> {
    // FIXME use a reference to `CodegenCx` instead of `tcx`, `module` and `constants` and `caches`
    pub tcx: TyCtxt<'tcx>,
    pub module: &'a mut Module<B>,
    pub pointer_type: Type, // Cached from module

    pub instance: Instance<'tcx>,
    pub mir: &'tcx Body<'tcx>,

    pub bcx: FunctionBuilder<'a>,
    pub ebb_map: HashMap<BasicBlock, Ebb>,
    pub local_map: HashMap<Local, CPlace<'tcx>>,

    pub clif_comments: crate::pretty_clif::CommentWriter,
    pub constants: &'a mut crate::constant::ConstantCx,
    pub caches: &'a mut Caches<'tcx>,
    pub source_info_set: indexmap::IndexSet<SourceInfo>,
}

impl<'a, 'tcx: 'a, B: Backend> LayoutOf for FunctionCx<'a, 'tcx, B> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> TyLayout<'tcx> {
        let ty = self.monomorphize(&ty);
        self.tcx.layout_of(ParamEnv::reveal_all().and(&ty))
            .unwrap_or_else(|e| if let layout::LayoutError::SizeOverflow(_) = e {
                self.tcx.sess.fatal(&e.to_string())
            } else {
                bug!("failed to get layout for `{}`: {}", ty, e)
            })
    }
}

impl<'a, 'tcx, B: Backend + 'a> layout::HasTyCtxt<'tcx> for FunctionCx<'a, 'tcx, B> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'a, 'tcx, B: Backend + 'a> layout::HasDataLayout for FunctionCx<'a, 'tcx, B> {
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'tcx, B: Backend + 'a> layout::HasParamEnv<'tcx> for FunctionCx<'a, 'tcx, B> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::reveal_all()
    }
}

impl<'a, 'tcx, B: Backend + 'a> HasTargetSpec for FunctionCx<'a, 'tcx, B> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target.target
    }
}

impl<'a, 'tcx, B: Backend> BackendTypes for FunctionCx<'a, 'tcx, B> {
    type Value = Value;
    type BasicBlock = Ebb;
    type Type = Type;
    type Funclet = !;
    type DIScope = !;
}

impl<'a, 'tcx: 'a, B: Backend + 'a> FunctionCx<'a, 'tcx, B> {
    pub fn monomorphize<T>(&self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.tcx.subst_and_normalize_erasing_regions(
            self.instance.substs,
            ty::ParamEnv::reveal_all(),
            value,
        )
    }

    pub fn clif_type(&self, ty: Ty<'tcx>) -> Option<Type> {
        clif_type_from_ty(self.tcx, self.monomorphize(&ty))
    }

    pub fn get_ebb(&self, bb: BasicBlock) -> Ebb {
        *self.ebb_map.get(&bb).unwrap()
    }

    pub fn get_local_place(&mut self, local: Local) -> CPlace<'tcx> {
        *self.local_map.get(&local).unwrap()
    }

    pub fn set_debug_loc(&mut self, source_info: mir::SourceInfo) {
        let (index, _) = self.source_info_set.insert_full(source_info);
        self.bcx.set_srcloc(SourceLoc::new(index as u32));
    }
}
