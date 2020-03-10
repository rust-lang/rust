use rustc::ty::layout::{Integer, Primitive};
use rustc_target::spec::{HasTargetSpec, Target};
use rustc_index::vec::IndexVec;

use cranelift_codegen::ir::{InstructionData, Opcode, ValueDef};

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

pub fn scalar_to_clif_type(tcx: TyCtxt, scalar: Scalar) -> Type {
    match scalar.value {
        Primitive::Int(int, _sign) => match int {
            Integer::I8 => types::I8,
            Integer::I16 => types::I16,
            Integer::I32 => types::I32,
            Integer::I64 => types::I64,
            Integer::I128 => types::I128,
        },
        Primitive::F32 => types::F32,
        Primitive::F64 => types::F64,
        Primitive::Pointer => pointer_ty(tcx),
    }
}

fn clif_type_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<types::Type> {
    Some(match ty.kind {
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
        ty::RawPtr(TypeAndMut { ty: pointee_ty, mutbl: _ }) | ty::Ref(_, pointee_ty, _) => {
            if has_ptr_meta(tcx, pointee_ty) {
                return None;
            } else {
                pointer_ty(tcx)
            }
        }
        ty::Param(_) => bug!("ty param {:?}", ty),
        _ => return None,
    })
}

/// Is a pointer to this type a fat ptr?
pub fn has_ptr_meta<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ptr_ty = tcx.mk_ptr(TypeAndMut { ty, mutbl: rustc_hir::Mutability::Not });
    match &tcx.layout_of(ParamEnv::reveal_all().and(ptr_ty)).unwrap().abi {
        Abi::Scalar(_) => false,
        Abi::ScalarPair(_, _) => true,
        abi => unreachable!("Abi of ptr to {:?} is {:?}???", ty, abi),
    }
}

pub fn codegen_icmp(
    fx: &mut FunctionCx<'_, '_, impl Backend>,
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

pub fn codegen_icmp_imm(
    fx: &mut FunctionCx<'_, '_, impl Backend>,
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
        } = func.dfg[inst]
        {
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
        } = func.dfg[inst]
        {
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
        (types::I8, false) | (types::I16, false) | (types::I32, false) | (types::I64, false) => {
            0i64
        }
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
    match ty.kind {
        ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::Char | ty::Uint(..) | ty::Bool => false,
        ty::Int(..) => true,
        ty::Float(..) => false, // `signed` is unused for floats
        _ => panic!("{}", ty),
    }
}

pub struct FunctionCx<'clif, 'tcx, B: Backend + 'static> {
    // FIXME use a reference to `CodegenCx` instead of `tcx`, `module` and `constants` and `caches`
    pub tcx: TyCtxt<'tcx>,
    pub module: &'clif mut Module<B>,
    pub pointer_type: Type, // Cached from module

    pub instance: Instance<'tcx>,
    pub mir: &'tcx Body<'tcx>,

    pub bcx: FunctionBuilder<'clif>,
    pub block_map: IndexVec<BasicBlock, Block>,
    pub local_map: HashMap<Local, CPlace<'tcx>>,

    /// When `#[track_caller]` is used, the implicit caller location is stored in this variable.
    pub caller_location: Option<CValue<'tcx>>,

    /// See [crate::optimize::code_layout] for more information.
    pub cold_blocks: EntitySet<Block>,

    pub clif_comments: crate::pretty_clif::CommentWriter,
    pub constants_cx: &'clif mut crate::constant::ConstantCx,
    pub vtables: &'clif mut HashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), DataId>,

    pub source_info_set: indexmap::IndexSet<SourceInfo>,
}

impl<'tcx, B: Backend> LayoutOf for FunctionCx<'_, 'tcx, B> {
    type Ty = Ty<'tcx>;
    type TyLayout = TyLayout<'tcx>;

    fn layout_of(&self, ty: Ty<'tcx>) -> TyLayout<'tcx> {
        assert!(!ty.needs_subst());
        self.tcx
            .layout_of(ParamEnv::reveal_all().and(&ty))
            .unwrap_or_else(|e| {
                if let layout::LayoutError::SizeOverflow(_) = e {
                    self.tcx.sess.fatal(&e.to_string())
                } else {
                    bug!("failed to get layout for `{}`: {}", ty, e)
                }
            })
    }
}

impl<'tcx, B: Backend + 'static> layout::HasTyCtxt<'tcx> for FunctionCx<'_, 'tcx, B> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, B: Backend + 'static> layout::HasDataLayout for FunctionCx<'_, 'tcx, B> {
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx, B: Backend + 'static> layout::HasParamEnv<'tcx> for FunctionCx<'_, 'tcx, B> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::reveal_all()
    }
}

impl<'tcx, B: Backend + 'static> HasTargetSpec for FunctionCx<'_, 'tcx, B> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target.target
    }
}

impl<'tcx, B: Backend> BackendTypes for FunctionCx<'_, 'tcx, B> {
    type Value = Value;
    type Function = Value;
    type BasicBlock = Block;
    type Type = Type;
    type Funclet = !;
    type DIScope = !;
    type DIVariable = !;
}

impl<'tcx, B: Backend + 'static> FunctionCx<'_, 'tcx, B> {
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
        clif_type_from_ty(self.tcx, ty)
    }

    pub fn get_block(&self, bb: BasicBlock) -> Block {
        *self.block_map.get(bb).unwrap()
    }

    pub fn get_local_place(&mut self, local: Local) -> CPlace<'tcx> {
        *self.local_map.get(&local).unwrap_or_else(|| {
            panic!("Local {:?} doesn't exist", local);
        })
    }

    pub fn set_debug_loc(&mut self, source_info: mir::SourceInfo) {
        let (index, _) = self.source_info_set.insert_full(source_info);
        self.bcx.set_srcloc(SourceLoc::new(index as u32));
    }

    pub fn get_caller_location(&mut self, span: Span) -> CValue<'tcx> {
        if let Some(loc) = self.caller_location {
            // `#[track_caller]` is used; return caller location instead of current location.
            return loc;
        }

        let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
        let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());
        let const_loc = self.tcx.const_caller_location((
            rustc_span::symbol::Symbol::intern(&caller.file.name.to_string()),
            caller.line as u32,
            caller.col_display as u32 + 1,
        ));
        crate::constant::trans_const_value(
            self,
            ty::Const::from_value(self.tcx, const_loc, self.tcx.caller_location_ty()),
        )
    }

    pub fn triple(&self) -> &target_lexicon::Triple {
        self.module.isa().triple()
    }
}
