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

pub fn type_min_max_value<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> (i64, i64) {
    use syntax::ast::UintTy::*;
    use syntax::ast::IntTy::*;

    let uint_usize_cvt = |uint| {
        match uint {
            UintTy::Usize => match pointer_ty(tcx) {
                types::I16 => UintTy::U16,
                types::I32 => UintTy::U32,
                types::I64 => UintTy::U64,
                ty => unreachable!("{:?}", ty),
            }
            _ => uint,
        }
    };

    let int_isize_cvt = |int| {
        match int {
            IntTy::Isize => match pointer_ty(tcx) {
                types::I16 => IntTy::I16,
                types::I32 => IntTy::I32,
                types::I64 => IntTy::I64,
                ty => unreachable!("{:?}", ty),
            }
            _ => int,
        }
    };

    let min = match ty.sty {
        ty::Uint(uint) => match uint_usize_cvt(uint) {
            U8 | U16 | U32 | U64 => 0i64,
            U128 => unimplemented!(),
            Usize => unreachable!(),
        }
        ty::Int(int) => match int_isize_cvt(int) {
            I8 => i8::min_value() as i64,
            I16 => i16::min_value() as i64,
            I32 => i32::min_value() as i64,
            I64 => i64::min_value(),
            I128 => unimplemented!(),
            Isize => unreachable!(),
        }
        _ => unreachable!(),
    };

    let max = match ty.sty {
        ty::Uint(uint) => match uint_usize_cvt(uint) {
            U8 => u8::max_value() as i64,
            U16 => u16::max_value() as i64,
            U32 => u32::max_value() as i64,
            U64 => u64::max_value() as i64,
            U128 => unimplemented!(),
            Usize => unreachable!(),
        }
        ty::Int(int) => match int_isize_cvt(int) {
            I8 => i8::max_value() as i64,
            I16 => i16::max_value() as i64,
            I32 => i32::max_value() as i64,
            I64 => i64::max_value(),
            I128 => unimplemented!(),
            Isize => unreachable!(),
        }
        _ => unreachable!(),
    };

    (min, max)
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
        self.tcx.layout_of(ParamEnv::reveal_all().and(&ty)).unwrap()
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
