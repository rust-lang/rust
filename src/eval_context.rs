use std::cell::Ref;
use std::collections::HashMap;

use rustc::hir::def_id::DefId;
use rustc::hir::map::definitions::DefPathData;
use rustc::middle::const_val::ConstVal;
use rustc::mir;
use rustc::traits::Reveal;
use rustc::ty::layout::{self, Layout, Size};
use rustc::ty::subst::{self, Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::fx::FxHashSet;
use syntax::codemap::{self, DUMMY_SP};

use error::{EvalError, EvalResult};
use lvalue::{Global, GlobalId, Lvalue, LvalueExtra};
use memory::{Memory, Pointer};
use operator;
use value::{PrimVal, PrimValKind, Value};

pub type MirRef<'tcx> = Ref<'tcx, mir::Mir<'tcx>>;

pub struct EvalContext<'a, 'tcx: 'a> {
    /// The results of the type checker, from rustc.
    pub(super) tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// The virtual memory system.
    pub(super) memory: Memory<'a, 'tcx>,

    /// Precomputed statics, constants and promoteds.
    pub(super) globals: HashMap<GlobalId<'tcx>, Global<'tcx>>,

    /// The virtual call stack.
    pub(super) stack: Vec<Frame<'tcx>>,

    /// The maximum number of stack frames allowed
    pub(super) stack_limit: usize,

    /// The maximum number of operations that may be executed.
    /// This prevents infinite loops and huge computations from freezing up const eval.
    /// Remove once halting problem is solved.
    pub(super) steps_remaining: u64,
}

/// A stack frame.
pub struct Frame<'tcx> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////

    /// The MIR for the function called on this frame.
    pub mir: MirRef<'tcx>,

    /// The def_id of the current function.
    pub def_id: DefId,

    /// type substitutions for the current function invocation.
    pub substs: &'tcx Substs<'tcx>,

    /// The span of the call site.
    pub span: codemap::Span,

    ////////////////////////////////////////////////////////////////////////////////
    // Return lvalue and locals
    ////////////////////////////////////////////////////////////////////////////////

    /// The block to return to when returning from the current stack frame
    pub return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to.
    pub return_lvalue: Lvalue<'tcx>,

    /// The list of locals for this stack frame, stored in order as
    /// `[arguments..., variables..., temporaries...]`. The locals are stored as `Value`s, which
    /// can either directly contain `PrimVal` or refer to some part of an `Allocation`.
    ///
    /// Before being initialized, all locals are `Value::ByVal(PrimVal::Undef)`.
    pub locals: Vec<Value>,

    /// Temporary allocations introduced to save stackframes
    /// This is pure interpreter magic and has nothing to do with how rustc does it
    /// An example is calling an FnMut closure that has been converted to a FnOnce closure
    /// The memory will be freed when the stackframe finishes
    pub interpreter_temporaries: Vec<Pointer>,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////

    /// The block that is currently executed (or will be executed after the above call stacks
    /// return).
    pub block: mir::BasicBlock,

    /// The index of the currently evaluated statment.
    pub stmt: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum StackPopCleanup {
    /// The stackframe existed to compute the initial value of a static/constant, make sure it
    /// isn't modifyable afterwards. The allocation of the result is frozen iff it's an
    /// actual allocation. `PrimVal`s are unmodifyable anyway.
    Freeze,
    /// A regular stackframe added due to a function call will need to get forwarded to the next
    /// block
    Goto(mir::BasicBlock),
    /// The main function and diverging functions have nowhere to return to
    None,
}

#[derive(Copy, Clone, Debug)]
pub struct ResourceLimits {
    pub memory_size: u64,
    pub step_limit: u64,
    pub stack_limit: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        ResourceLimits {
            memory_size: 100 * 1024 * 1024, // 100 MB
            step_limit: 1_000_000,
            stack_limit: 100,
        }
    }
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, limits: ResourceLimits) -> Self {
        EvalContext {
            tcx: tcx,
            memory: Memory::new(&tcx.data_layout, limits.memory_size),
            globals: HashMap::new(),
            stack: Vec::new(),
            stack_limit: limits.stack_limit,
            steps_remaining: limits.step_limit,
        }
    }

    pub fn alloc_ptr(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, Pointer> {
        let substs = self.substs();
        self.alloc_ptr_with_substs(ty, substs)
    }

    pub fn alloc_ptr_with_substs(
        &mut self,
        ty: Ty<'tcx>,
        substs: &'tcx Substs<'tcx>
    ) -> EvalResult<'tcx, Pointer> {
        let size = self.type_size_with_substs(ty, substs)?.expect("cannot alloc memory for unsized type");
        let align = self.type_align_with_substs(ty, substs)?;
        self.memory.allocate(size, align)
    }

    pub fn memory(&self) -> &Memory<'a, 'tcx> {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx> {
        &mut self.memory
    }

    pub fn stack(&self) -> &[Frame<'tcx>] {
        &self.stack
    }

    pub(super) fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        // FIXME: cache these allocs
        let ptr = self.memory.allocate(s.len() as u64, 1)?;
        self.memory.write_bytes(ptr, s.as_bytes())?;
        self.memory.freeze(ptr.alloc_id)?;
        Ok(Value::ByValPair(PrimVal::Ptr(ptr), PrimVal::from_u128(s.len() as u128)))
    }

    pub(super) fn const_to_value(&mut self, const_val: &ConstVal) -> EvalResult<'tcx, Value> {
        use rustc::middle::const_val::ConstVal::*;
        use rustc_const_math::ConstFloat;

        let primval = match *const_val {
            Integral(const_int) => PrimVal::Bytes(const_int.to_u128_unchecked()),

            Float(ConstFloat::F32(f)) => PrimVal::from_f32(f),
            Float(ConstFloat::F64(f)) => PrimVal::from_f64(f),
            Float(ConstFloat::FInfer { .. }) =>
                bug!("uninferred constants only exist before typeck"),

            Bool(b) => PrimVal::from_bool(b),
            Char(c) => PrimVal::from_char(c),

            Str(ref s) => return self.str_to_value(s),

            ByteStr(ref bs) => {
                let ptr = self.memory.allocate(bs.len() as u64, 1)?;
                self.memory.write_bytes(ptr, bs)?;
                self.memory.freeze(ptr.alloc_id)?;
                PrimVal::Ptr(ptr)
            }

            Struct(_)    => unimplemented!(),
            Tuple(_)     => unimplemented!(),
            Function(_)  => unimplemented!(),
            Array(_, _)  => unimplemented!(),
            Repeat(_, _) => unimplemented!(),
            Dummy        => unimplemented!(),
        };

        Ok(Value::ByVal(primval))
    }

    pub(super) fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        // generics are weird, don't run this function on a generic
        assert!(!ty.needs_subst());
        ty.is_sized(self.tcx, &self.tcx.empty_parameter_environment(), DUMMY_SP)
    }

    pub fn load_mir(&self, def_id: DefId) -> EvalResult<'tcx, MirRef<'tcx>> {
        trace!("load mir {:?}", def_id);
        if def_id.is_local() || self.tcx.sess.cstore.is_item_mir_available(def_id) {
            Ok(self.tcx.item_mir(def_id))
        } else {
            Err(EvalError::NoMirFor(self.tcx.item_path_str(def_id)))
        }
    }

    pub fn monomorphize(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        let substituted = ty.subst(self.tcx, substs);
        self.tcx.normalize_associated_type(&substituted)
    }

    pub(super) fn type_size(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<u64>> {
        self.type_size_with_substs(ty, self.substs())
    }

    pub(super) fn type_align(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, u64> {
        self.type_align_with_substs(ty, self.substs())
    }

    fn type_size_with_substs(
        &self,
        ty: Ty<'tcx>,
        substs: &'tcx Substs<'tcx>,
    ) -> EvalResult<'tcx, Option<u64>> {
        let layout = self.type_layout_with_substs(ty, substs)?;
        if layout.is_unsized() {
            Ok(None)
        } else {
            Ok(Some(layout.size(&self.tcx.data_layout).bytes()))
        }
    }

    fn type_align_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, u64> {
        self.type_layout_with_substs(ty, substs).map(|layout| layout.align(&self.tcx.data_layout).abi())
    }

    pub(super) fn type_layout(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, &'tcx Layout> {
        self.type_layout_with_substs(ty, self.substs())
    }

    fn type_layout_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, &'tcx Layout> {
        // TODO(solson): Is this inefficient? Needs investigation.
        let ty = self.monomorphize(ty, substs);

        self.tcx.infer_ctxt(None, None, Reveal::All).enter(|infcx| {
            ty.layout(&infcx).map_err(EvalError::Layout)
        })
    }

    pub fn push_stack_frame(
        &mut self,
        def_id: DefId,
        span: codemap::Span,
        mir: MirRef<'tcx>,
        substs: &'tcx Substs<'tcx>,
        return_lvalue: Lvalue<'tcx>,
        return_to_block: StackPopCleanup,
        temporaries: Vec<Pointer>,
    ) -> EvalResult<'tcx, ()> {
        ::log_settings::settings().indentation += 1;

        // Subtract 1 because `local_decls` includes the ReturnPointer, but we don't store a local
        // `Value` for that.
        let num_locals = mir.local_decls.len() - 1;
        let locals = vec![Value::ByVal(PrimVal::Undef); num_locals];

        self.stack.push(Frame {
            mir: mir,
            block: mir::START_BLOCK,
            return_to_block: return_to_block,
            return_lvalue: return_lvalue,
            locals: locals,
            interpreter_temporaries: temporaries,
            span: span,
            def_id: def_id,
            substs: substs,
            stmt: 0,
        });

        if self.stack.len() > self.stack_limit {
            Err(EvalError::StackFrameLimitReached)
        } else {
            Ok(())
        }
    }

    pub(super) fn pop_stack_frame(&mut self) -> EvalResult<'tcx, ()> {
        ::log_settings::settings().indentation -= 1;
        let frame = self.stack.pop().expect("tried to pop a stack frame, but there were none");
        match frame.return_to_block {
            StackPopCleanup::Freeze => if let Lvalue::Global(id) = frame.return_lvalue {
                let global_value = self.globals.get_mut(&id)
                    .expect("global should have been cached (freeze)");
                match global_value.value {
                    Value::ByRef(ptr) => self.memory.freeze(ptr.alloc_id)?,
                    Value::ByVal(val) => if let PrimVal::Ptr(ptr) = val {
                        self.memory.freeze(ptr.alloc_id)?;
                    },
                    Value::ByValPair(val1, val2) => {
                        if let PrimVal::Ptr(ptr) = val1 {
                            self.memory.freeze(ptr.alloc_id)?;
                        }
                        if let PrimVal::Ptr(ptr) = val2 {
                            self.memory.freeze(ptr.alloc_id)?;
                        }
                    },
                }
                assert!(global_value.mutable);
                global_value.mutable = false;
            } else {
                bug!("StackPopCleanup::Freeze on: {:?}", frame.return_lvalue);
            },
            StackPopCleanup::Goto(target) => self.goto_block(target),
            StackPopCleanup::None => {},
        }
        // deallocate all locals that are backed by an allocation
        for local in frame.locals.into_iter() {
            if let Value::ByRef(ptr) = local {
                trace!("deallocating local");
                self.memory.dump_alloc(ptr.alloc_id);
                match self.memory.deallocate(ptr) {
                    // Any frozen memory means that it belongs to a constant or something referenced
                    // by a constant. We could alternatively check whether the alloc_id is frozen
                    // before calling deallocate, but this is much simpler and is probably the
                    // rare case.
                    Ok(()) | Err(EvalError::DeallocatedFrozenMemory) => {},
                    other => return other,
                }
            }
        }
        // deallocate all temporary allocations
        for ptr in frame.interpreter_temporaries {
            trace!("deallocating temporary allocation");
            self.memory.dump_alloc(ptr.alloc_id);
            self.memory.deallocate(ptr)?;
        }
        Ok(())
    }

    fn assign_fields<I: IntoIterator<Item = u64>>(
        &mut self,
        dest: Lvalue<'tcx>,
        offsets: I,
        operands: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx, ()> {
        // FIXME(solson)
        let dest = self.force_allocation(dest)?.to_ptr();

        for (offset, operand) in offsets.into_iter().zip(operands) {
            let value = self.eval_operand(operand)?;
            let value_ty = self.operand_ty(operand);
            let field_dest = dest.offset(offset);
            self.write_value_to_ptr(value, field_dest, value_ty)?;
        }
        Ok(())
    }

    /// Evaluate an assignment statement.
    ///
    /// There is no separate `eval_rvalue` function. Instead, the code for handling each rvalue
    /// type writes its results directly into the memory specified by the lvalue.
    pub(super) fn eval_rvalue_into_lvalue(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        lvalue: &mir::Lvalue<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        let dest = self.eval_lvalue(lvalue)?;
        let dest_ty = self.lvalue_ty(lvalue);
        let dest_layout = self.type_layout(dest_ty)?;

        use rustc::mir::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let value = self.eval_operand(operand)?;
                self.write_value(value, dest, dest_ty)?;
            }

            BinaryOp(bin_op, ref left, ref right) => {
                // ignore overflow bit, rustc inserts check branches for us
                self.intrinsic_overflowing(bin_op, left, right, dest, dest_ty)?;
            }

            CheckedBinaryOp(bin_op, ref left, ref right) => {
                self.intrinsic_with_overflow(bin_op, left, right, dest, dest_ty)?;
            }

            UnaryOp(un_op, ref operand) => {
                let val = self.eval_operand_to_primval(operand)?;
                let kind = self.ty_to_primval_kind(dest_ty)?;
                self.write_primval(dest, operator::unary_op(un_op, val, kind)?, dest_ty)?;
            }

            Aggregate(ref kind, ref operands) => {
                self.inc_step_counter_and_check_limit(operands.len() as u64)?;
                use rustc::ty::layout::Layout::*;
                match *dest_layout {
                    Univariant { ref variant, .. } => {
                        let offsets = variant.offsets.iter().map(|s| s.bytes());
                        self.assign_fields(dest, offsets, operands)?;
                    }

                    Array { .. } => {
                        let elem_size = match dest_ty.sty {
                            ty::TyArray(elem_ty, _) => self.type_size(elem_ty)?
                                .expect("array elements are sized") as u64,
                            _ => bug!("tried to assign {:?} to non-array type {:?}", kind, dest_ty),
                        };
                        let offsets = (0..).map(|i| i * elem_size);
                        self.assign_fields(dest, offsets, operands)?;
                    }

                    General { discr, ref variants, .. } => {
                        if let mir::AggregateKind::Adt(adt_def, variant, _, _) = *kind {
                            let discr_val = adt_def.variants[variant].disr_val.to_u128_unchecked();
                            let discr_size = discr.size().bytes();
                            let discr_offset = variants[variant].offsets[0].bytes();

                            // FIXME(solson)
                            let dest = self.force_allocation(dest)?;
                            let discr_dest = (dest.to_ptr()).offset(discr_offset);

                            self.memory.write_uint(discr_dest, discr_val, discr_size)?;

                            // Don't include the first offset; it's for the discriminant.
                            let field_offsets = variants[variant].offsets.iter().skip(1)
                                .map(|s| s.bytes());
                            self.assign_fields(dest, field_offsets, operands)?;
                        } else {
                            bug!("tried to assign {:?} to Layout::General", kind);
                        }
                    }

                    RawNullablePointer { nndiscr, .. } => {
                        if let mir::AggregateKind::Adt(_, variant, _, _) = *kind {
                            if nndiscr == variant as u64 {
                                assert_eq!(operands.len(), 1);
                                let operand = &operands[0];
                                let value = self.eval_operand(operand)?;
                                let value_ty = self.operand_ty(operand);
                                self.write_value(value, dest, value_ty)?;
                            } else {
                                if let Some(operand) = operands.get(0) {
                                    assert_eq!(operands.len(), 1);
                                    let operand_ty = self.operand_ty(operand);
                                    assert_eq!(self.type_size(operand_ty)?, Some(0));
                                }
                                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                            }
                        } else {
                            bug!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    StructWrappedNullablePointer { nndiscr, ref nonnull, ref discrfield, .. } => {
                        if let mir::AggregateKind::Adt(_, variant, _, _) = *kind {
                            if nndiscr == variant as u64 {
                                let offsets = nonnull.offsets.iter().map(|s| s.bytes());
                                self.assign_fields(dest, offsets, operands)?;
                            } else {
                                for operand in operands {
                                    let operand_ty = self.operand_ty(operand);
                                    assert_eq!(self.type_size(operand_ty)?, Some(0));
                                }
                                let (offset, ty) = self.nonnull_offset_and_ty(dest_ty, nndiscr, discrfield)?;

                                // FIXME(solson)
                                let dest = self.force_allocation(dest)?.to_ptr();

                                let dest = dest.offset(offset.bytes());
                                let dest_size = self.type_size(ty)?
                                    .expect("bad StructWrappedNullablePointer discrfield");
                                self.memory.write_int(dest, 0, dest_size)?;
                            }
                        } else {
                            bug!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    CEnum { .. } => {
                        assert_eq!(operands.len(), 0);
                        if let mir::AggregateKind::Adt(adt_def, variant, _, _) = *kind {
                            let n = adt_def.variants[variant].disr_val.to_u128_unchecked();
                            self.write_primval(dest, PrimVal::Bytes(n), dest_ty)?;
                        } else {
                            bug!("tried to assign {:?} to Layout::CEnum", kind);
                        }
                    }

                    Vector { element, count } => {
                        let elem_size = element.size(&self.tcx.data_layout).bytes();
                        debug_assert_eq!(count, operands.len() as u64);
                        let offsets = (0..).map(|i| i * elem_size);
                        self.assign_fields(dest, offsets, operands)?;
                    }

                    UntaggedUnion { .. } => {
                        assert_eq!(operands.len(), 1);
                        let operand = &operands[0];
                        let value = self.eval_operand(operand)?;
                        let value_ty = self.operand_ty(operand);
                        self.write_value(value, dest, value_ty)?;
                    }

                    _ => {
                        return Err(EvalError::Unimplemented(format!(
                            "can't handle destination layout {:?} when assigning {:?}",
                            dest_layout,
                            kind
                        )));
                    }
                }
            }

            Repeat(ref operand, _) => {
                let (elem_ty, length) = match dest_ty.sty {
                    ty::TyArray(elem_ty, n) => (elem_ty, n as u64),
                    _ => bug!("tried to assign array-repeat to non-array type {:?}", dest_ty),
                };
                self.inc_step_counter_and_check_limit(length)?;
                let elem_size = self.type_size(elem_ty)?
                    .expect("repeat element type must be sized");
                let value = self.eval_operand(operand)?;

                // FIXME(solson)
                let dest = self.force_allocation(dest)?.to_ptr();

                for i in 0..length {
                    let elem_dest = dest.offset(i * elem_size);
                    self.write_value_to_ptr(value, elem_dest, elem_ty)?;
                }
            }

            Len(ref lvalue) => {
                let src = self.eval_lvalue(lvalue)?;
                let ty = self.lvalue_ty(lvalue);
                let (_, len) = src.elem_ty_and_len(ty);
                self.write_primval(dest, PrimVal::from_u128(len as u128), dest_ty)?;
            }

            Ref(_, _, ref lvalue) => {
                let src = self.eval_lvalue(lvalue)?;
                let (raw_ptr, extra) = self.force_allocation(src)?.to_ptr_and_extra();
                let ptr = PrimVal::Ptr(raw_ptr);

                let val = match extra {
                    LvalueExtra::None => Value::ByVal(ptr),
                    LvalueExtra::Length(len) => Value::ByValPair(ptr, PrimVal::from_u128(len as u128)),
                    LvalueExtra::Vtable(vtable) => Value::ByValPair(ptr, PrimVal::Ptr(vtable)),
                    LvalueExtra::DowncastVariant(..) =>
                        bug!("attempted to take a reference to an enum downcast lvalue"),
                };

                self.write_value(val, dest, dest_ty)?;
            }

            Box(ty) => {
                let ptr = self.alloc_ptr(ty)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }

            Cast(kind, ref operand, cast_ty) => {
                debug_assert_eq!(self.monomorphize(cast_ty, self.substs()), dest_ty);
                use rustc::mir::CastKind::*;
                match kind {
                    Unsize => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        self.unsize_into(src, src_ty, dest, dest_ty)?;
                    }

                    Misc => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        if self.type_is_fat_ptr(src_ty) {
                            trace!("misc cast: {:?}", src);
                            match (src, self.type_is_fat_ptr(dest_ty)) {
                                (Value::ByRef(_), _) |
                                (Value::ByValPair(..), true) => {
                                    self.write_value(src, dest, dest_ty)?;
                                },
                                (Value::ByValPair(data, _), false) => {
                                    self.write_value(Value::ByVal(data), dest, dest_ty)?;
                                },
                                (Value::ByVal(_), _) => bug!("expected fat ptr"),
                            }
                        } else {
                            let src_val = self.value_to_primval(src, src_ty)?;
                            let dest_val = self.cast_primval(src_val, src_ty, dest_ty)?;
                            self.write_value(Value::ByVal(dest_val), dest, dest_ty)?;
                        }
                    }

                    ReifyFnPointer => match self.operand_ty(operand).sty {
                        ty::TyFnDef(def_id, substs, fn_ty) => {
                            let fn_ty = self.tcx.erase_regions(&fn_ty);
                            let fn_ptr = self.memory.create_fn_ptr(self.tcx,def_id, substs, fn_ty);
                            self.write_value(Value::ByVal(PrimVal::Ptr(fn_ptr)), dest, dest_ty)?;
                        },
                        ref other => bug!("reify fn pointer on {:?}", other),
                    },

                    UnsafeFnPointer => match dest_ty.sty {
                        ty::TyFnPtr(unsafe_fn_ty) => {
                            let src = self.eval_operand(operand)?;
                            let ptr = src.read_ptr(&self.memory)?;
                            let (def_id, substs, _, _) = self.memory.get_fn(ptr.alloc_id)?;
                            let unsafe_fn_ty = self.tcx.erase_regions(&unsafe_fn_ty);
                            let fn_ptr = self.memory.create_fn_ptr(self.tcx, def_id, substs, unsafe_fn_ty);
                            self.write_value(Value::ByVal(PrimVal::Ptr(fn_ptr)), dest, dest_ty)?;
                        },
                        ref other => bug!("fn to unsafe fn cast on {:?}", other),
                    },
                }
            }

            InlineAsm { .. } => return Err(EvalError::InlineAsm),
        }

        if log_enabled!(::log::LogLevel::Trace) {
            self.dump_local(dest);
        }

        Ok(())
    }

    fn type_is_fat_ptr(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyRawPtr(ty::TypeAndMut{ty, ..}) |
            ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
            ty::TyBox(ty) => !self.type_is_sized(ty),
            _ => false,
        }
    }

    pub(super) fn nonnull_offset_and_ty(
        &self,
        ty: Ty<'tcx>,
        nndiscr: u64,
        discrfield: &[u32],
    ) -> EvalResult<'tcx, (Size, Ty<'tcx>)> {
        // Skip the constant 0 at the start meant for LLVM GEP and the outer non-null variant
        let path = discrfield.iter().skip(2).map(|&i| i as usize);

        // Handle the field index for the outer non-null variant.
        let inner_ty = match ty.sty {
            ty::TyAdt(adt_def, substs) => {
                let variant = &adt_def.variants[nndiscr as usize];
                let index = discrfield[1];
                let field = &variant.fields[index as usize];
                field.ty(self.tcx, substs)
            }
            _ => bug!("non-enum for StructWrappedNullablePointer: {}", ty),
        };

        self.field_path_offset_and_ty(inner_ty, path)
    }

    fn field_path_offset_and_ty<I: Iterator<Item = usize>>(&self, mut ty: Ty<'tcx>, path: I) -> EvalResult<'tcx, (Size, Ty<'tcx>)> {
        let mut offset = Size::from_bytes(0);

        // Skip the initial 0 intended for LLVM GEP.
        for field_index in path {
            let field_offset = self.get_field_offset(ty, field_index)?;
            trace!("field_path_offset_and_ty: {}, {}, {:?}, {:?}", field_index, ty, field_offset, offset);
            ty = self.get_field_ty(ty, field_index)?;
            offset = offset.checked_add(field_offset, &self.tcx.data_layout).unwrap();
        }

        Ok((offset, ty))
    }

    pub fn get_field_ty(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<'tcx, Ty<'tcx>> {
        match ty.sty {
            ty::TyAdt(adt_def, substs) => {
                Ok(adt_def.struct_variant().fields[field_index].ty(self.tcx, substs))
            }

            ty::TyTuple(fields) => Ok(fields[field_index]),

            ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
            ty::TyBox(ty) => {
                match (field_index, &self.tcx.struct_tail(ty).sty) {
                    (1, &ty::TyStr) |
                    (1, &ty::TySlice(_)) => Ok(self.tcx.types.usize),
                    (1, &ty::TyDynamic(..)) |
                    (0, _) => Ok(self.tcx.mk_imm_ptr(self.tcx.types.u8)),
                    _ => bug!("invalid fat pointee type: {}", ty),
                }
            }
            _ => Err(EvalError::Unimplemented(format!("can't handle type: {:?}, {:?}", ty, ty.sty))),
        }
    }

    fn get_field_offset(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<'tcx, Size> {
        let layout = self.type_layout(ty)?;

        use rustc::ty::layout::Layout::*;
        match *layout {
            Univariant { ref variant, .. } => {
                Ok(variant.offsets[field_index])
            }
            FatPointer { .. } => {
                let bytes = field_index as u64 * self.memory.pointer_size();
                Ok(Size::from_bytes(bytes))
            }
            _ => {
                let msg = format!("can't handle type: {:?}, with layout: {:?}", ty, layout);
                Err(EvalError::Unimplemented(msg))
            }
        }
    }

    fn get_field_count(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, usize> {
        let layout = self.type_layout(ty)?;

        use rustc::ty::layout::Layout::*;
        match *layout {
            Univariant { ref variant, .. } => Ok(variant.offsets.len()),
            FatPointer { .. } => Ok(2),
            _ => {
                let msg = format!("can't handle type: {:?}, with layout: {:?}", ty, layout);
                Err(EvalError::Unimplemented(msg))
            }
        }
    }

    pub(super) fn eval_operand_to_primval(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, PrimVal> {
        let value = self.eval_operand(op)?;
        let ty = self.operand_ty(op);
        self.value_to_primval(value, ty)
    }

    pub(super) fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, Value> {
        use rustc::mir::Operand::*;
        match *op {
            Consume(ref lvalue) => self.eval_and_read_lvalue(lvalue),

            Constant(mir::Constant { ref literal, ty, .. }) => {
                use rustc::mir::Literal;
                let value = match *literal {
                    Literal::Value { ref value } => self.const_to_value(value)?,

                    Literal::Item { def_id, substs } => {
                        if let ty::TyFnDef(..) = ty.sty {
                            // function items are zero sized
                            Value::ByRef(self.memory.allocate(0, 0)?)
                        } else {
                            let cid = GlobalId {
                                def_id: def_id,
                                substs: substs,
                                promoted: None,
                            };
                            self.read_lvalue(Lvalue::Global(cid))
                        }
                    }

                    Literal::Promoted { index } => {
                        let cid = GlobalId {
                            def_id: self.frame().def_id,
                            substs: self.substs(),
                            promoted: Some(index),
                        };
                        self.read_lvalue(Lvalue::Global(cid))
                    }
                };

                Ok(value)
            }
        }
    }

    pub(super) fn operand_ty(&self, operand: &mir::Operand<'tcx>) -> Ty<'tcx> {
        self.monomorphize(operand.ty(&self.mir(), self.tcx), self.substs())
    }

    fn copy(&mut self, src: Pointer, dest: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, ()> {
        let size = self.type_size(ty)?.expect("cannot copy from an unsized type");
        let align = self.type_align(ty)?;
        self.memory.copy(src, dest, size, align)?;
        Ok(())
    }

    pub(super) fn force_allocation(
        &mut self,
        lvalue: Lvalue<'tcx>,
    ) -> EvalResult<'tcx, Lvalue<'tcx>> {
        let new_lvalue = match lvalue {
            Lvalue::Local { frame, local } => {
                match self.stack[frame].get_local(local) {
                    Value::ByRef(ptr) => Lvalue::from_ptr(ptr),
                    val => {
                        let ty = self.stack[frame].mir.local_decls[local].ty;
                        let ty = self.monomorphize(ty, self.stack[frame].substs);
                        let substs = self.stack[frame].substs;
                        let ptr = self.alloc_ptr_with_substs(ty, substs)?;
                        self.stack[frame].set_local(local, Value::ByRef(ptr));
                        self.write_value_to_ptr(val, ptr, ty)?;
                        Lvalue::from_ptr(ptr)
                    }
                }
            }
            Lvalue::Ptr { .. } => lvalue,
            Lvalue::Global(cid) => {
                let global_val = *self.globals.get(&cid).expect("global not cached");
                match global_val.value {
                    Value::ByRef(ptr) => Lvalue::from_ptr(ptr),
                    _ => {
                        let ptr = self.alloc_ptr_with_substs(global_val.ty, cid.substs)?;
                        self.write_value_to_ptr(global_val.value, ptr, global_val.ty)?;
                        if !global_val.mutable {
                            self.memory.freeze(ptr.alloc_id)?;
                        }
                        let lval = self.globals.get_mut(&cid).expect("already checked");
                        *lval = Global {
                            value: Value::ByRef(ptr),
                            .. global_val
                        };
                        Lvalue::from_ptr(ptr)
                    },
                }
            }
        };
        Ok(new_lvalue)
    }

    /// ensures this Value is not a ByRef
    pub(super) fn follow_by_ref_value(&mut self, value: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        match value {
            Value::ByRef(ptr) => self.read_value(ptr, ty),
            other => Ok(other),
        }
    }

    pub(super) fn value_to_primval(&mut self, value: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        match self.follow_by_ref_value(value, ty)? {
            Value::ByRef(_) => bug!("follow_by_ref_value can't result in `ByRef`"),

            Value::ByVal(primval) => {
                self.ensure_valid_value(primval, ty)?;
                Ok(primval)
            }

            Value::ByValPair(..) => bug!("value_to_primval can't work with fat pointers"),
        }
    }

    pub(super) fn write_primval(
        &mut self,
        dest: Lvalue<'tcx>,
        val: PrimVal,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        match dest {
            Lvalue::Ptr { ptr, extra } => {
                assert_eq!(extra, LvalueExtra::None);
                let size = self.type_size(dest_ty)?.expect("dest type must be sized");
                self.memory.write_primval(ptr, val, size)
            }
            Lvalue::Local { frame, local } => {
                self.stack[frame].set_local(local, Value::ByVal(val));
                Ok(())
            }
            Lvalue::Global(cid) => {
                let global_val = self.globals.get_mut(&cid).expect("global not cached");
                if global_val.mutable {
                    global_val.value = Value::ByVal(val);
                    Ok(())
                } else {
                    Err(EvalError::ModifiedConstantMemory)
                }
            }
        }
    }

    pub(super) fn write_value(
        &mut self,
        src_val: Value,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        match dest {
            Lvalue::Global(cid) => {
                let dest = *self.globals.get_mut(&cid).expect("global should be cached");
                if !dest.mutable {
                    return Err(EvalError::ModifiedConstantMemory);
                }
                let write_dest = |this: &mut Self, val| {
                    *this.globals.get_mut(&cid).expect("already checked") = Global {
                        value: val,
                        ..dest
                    }
                };
                self.write_value_possibly_by_val(src_val, write_dest, dest.value, dest_ty)
            },

            Lvalue::Ptr { ptr, extra } => {
                assert_eq!(extra, LvalueExtra::None);
                self.write_value_to_ptr(src_val, ptr, dest_ty)
            }

            Lvalue::Local { frame, local } => {
                let dest = self.stack[frame].get_local(local);
                self.write_value_possibly_by_val(
                    src_val,
                    |this, val| this.stack[frame].set_local(local, val),
                    dest,
                    dest_ty,
                )
            }
        }
    }

    // The cases here can be a bit subtle. Read carefully!
    fn write_value_possibly_by_val<F: FnOnce(&mut Self, Value)>(
        &mut self,
        src_val: Value,
        write_dest: F,
        old_dest_val: Value,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        if let Value::ByRef(dest_ptr) = old_dest_val {
            // If the value is already `ByRef` (that is, backed by an `Allocation`),
            // then we must write the new value into this allocation, because there may be
            // other pointers into the allocation. These other pointers are logically
            // pointers into the local variable, and must be able to observe the change.
            //
            // Thus, it would be an error to replace the `ByRef` with a `ByVal`, unless we
            // knew for certain that there were no outstanding pointers to this allocation.
            self.write_value_to_ptr(src_val, dest_ptr, dest_ty)?;

        } else if let Value::ByRef(src_ptr) = src_val {
            // If the value is not `ByRef`, then we know there are no pointers to it
            // and we can simply overwrite the `Value` in the locals array directly.
            //
            // In this specific case, where the source value is `ByRef`, we must duplicate
            // the allocation, because this is a by-value operation. It would be incorrect
            // if they referred to the same allocation, since then a change to one would
            // implicitly change the other.
            //
            // It is a valid optimization to attempt reading a primitive value out of the
            // source and write that into the destination without making an allocation, so
            // we do so here.
            if let Ok(Some(src_val)) = self.try_read_value(src_ptr, dest_ty) {
                write_dest(self, src_val);
            } else {
                let dest_ptr = self.alloc_ptr(dest_ty)?;
                self.copy(src_ptr, dest_ptr, dest_ty)?;
                write_dest(self, Value::ByRef(dest_ptr));
            }

        } else {
            // Finally, we have the simple case where neither source nor destination are
            // `ByRef`. We may simply copy the source value over the the destintion.
            write_dest(self, src_val);
        }
        Ok(())
    }

    pub(super) fn write_value_to_ptr(
        &mut self,
        value: Value,
        dest: Pointer,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        match value {
            Value::ByRef(ptr) => self.copy(ptr, dest, dest_ty),
            Value::ByVal(primval) => {
                let size = self.type_size(dest_ty)?.expect("dest type must be sized");
                self.memory.write_primval(dest, primval, size)
            }
            Value::ByValPair(a, b) => self.write_pair_to_ptr(a, b, dest, dest_ty),
        }
    }

    pub(super) fn write_pair_to_ptr(
        &mut self,
        a: PrimVal,
        b: PrimVal,
        ptr: Pointer,
        ty: Ty<'tcx>
    ) -> EvalResult<'tcx, ()> {
        assert_eq!(self.get_field_count(ty)?, 2);
        let field_0 = self.get_field_offset(ty, 0)?.bytes();
        let field_1 = self.get_field_offset(ty, 1)?.bytes();
        let field_0_ty = self.get_field_ty(ty, 0)?;
        let field_1_ty = self.get_field_ty(ty, 1)?;
        let field_0_size = self.type_size(field_0_ty)?.expect("pair element type must be sized");
        let field_1_size = self.type_size(field_1_ty)?.expect("pair element type must be sized");
        self.memory.write_primval(ptr.offset(field_0), a, field_0_size)?;
        self.memory.write_primval(ptr.offset(field_1), b, field_1_size)?;
        Ok(())
    }

    pub fn ty_to_primval_kind(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimValKind> {
        use syntax::ast::FloatTy;

        let kind = match ty.sty {
            ty::TyBool => PrimValKind::Bool,
            ty::TyChar => PrimValKind::Char,

            ty::TyInt(int_ty) => {
                use syntax::ast::IntTy::*;
                let size = match int_ty {
                    I8 => 1,
                    I16 => 2,
                    I32 => 4,
                    I64 => 8,
                    I128 => 16,
                    Is => self.memory.pointer_size(),
                };
                PrimValKind::from_int_size(size)
            }

            ty::TyUint(uint_ty) => {
                use syntax::ast::UintTy::*;
                let size = match uint_ty {
                    U8 => 1,
                    U16 => 2,
                    U32 => 4,
                    U64 => 8,
                    U128 => 16,
                    Us => self.memory.pointer_size(),
                };
                PrimValKind::from_uint_size(size)
            }

            ty::TyFloat(FloatTy::F32) => PrimValKind::F32,
            ty::TyFloat(FloatTy::F64) => PrimValKind::F64,

            ty::TyFnPtr(_) => PrimValKind::FnPtr,

            ty::TyBox(ty) |
            ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) if self.type_is_sized(ty) => PrimValKind::Ptr,

            ty::TyAdt(..) => {
                use rustc::ty::layout::Layout::*;
                match *self.type_layout(ty)? {
                    CEnum { discr, signed, .. } => {
                        let size = discr.size().bytes();
                        if signed {
                            PrimValKind::from_int_size(size)
                        } else {
                            PrimValKind::from_uint_size(size)
                        }
                    }

                    RawNullablePointer { value, .. } => {
                        use rustc::ty::layout::Primitive::*;
                        match value {
                            // TODO(solson): Does signedness matter here? What should the sign be?
                            Int(int) => PrimValKind::from_uint_size(int.size().bytes()),
                            F32 => PrimValKind::F32,
                            F64 => PrimValKind::F64,
                            Pointer => PrimValKind::Ptr,
                        }
                    }

                    _ => return Err(EvalError::TypeNotPrimitive(ty)),
                }
            }

            _ => return Err(EvalError::TypeNotPrimitive(ty)),
        };

        Ok(kind)
    }

    fn ensure_valid_value(&self, val: PrimVal, ty: Ty<'tcx>) -> EvalResult<'tcx, ()> {
        match ty.sty {
            ty::TyBool if val.to_bytes()? > 1 => Err(EvalError::InvalidBool),

            ty::TyChar if ::std::char::from_u32(val.to_bytes()? as u32).is_none()
                => Err(EvalError::InvalidChar(val.to_bytes()? as u32 as u128)),

            _ => Ok(()),
        }
    }

    pub(super) fn read_value(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    fn try_read_value(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        use syntax::ast::FloatTy;

        let val = match ty.sty {
            ty::TyBool => PrimVal::from_bool(self.memory.read_bool(ptr)?),
            ty::TyChar => {
                let c = self.memory.read_uint(ptr, 4)? as u32;
                match ::std::char::from_u32(c) {
                    Some(ch) => PrimVal::from_char(ch),
                    None => return Err(EvalError::InvalidChar(c as u128)),
                }
            }

            ty::TyInt(int_ty) => {
                use syntax::ast::IntTy::*;
                let size = match int_ty {
                    I8 => 1,
                    I16 => 2,
                    I32 => 4,
                    I64 => 8,
                    I128 => 16,
                    Is => self.memory.pointer_size(),
                };
                PrimVal::from_i128(self.memory.read_int(ptr, size)?)
            }

            ty::TyUint(uint_ty) => {
                use syntax::ast::UintTy::*;
                let size = match uint_ty {
                    U8 => 1,
                    U16 => 2,
                    U32 => 4,
                    U64 => 8,
                    U128 => 16,
                    Us => self.memory.pointer_size(),
                };
                PrimVal::from_u128(self.memory.read_uint(ptr, size)?)
            }

            ty::TyFloat(FloatTy::F32) => PrimVal::from_f32(self.memory.read_f32(ptr)?),
            ty::TyFloat(FloatTy::F64) => PrimVal::from_f64(self.memory.read_f64(ptr)?),

            ty::TyFnPtr(_) => self.memory.read_ptr(ptr).map(PrimVal::Ptr)?,
            ty::TyBox(ty) |
            ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) => {
                let p = self.memory.read_ptr(ptr)?;
                if self.type_is_sized(ty) {
                    PrimVal::Ptr(p)
                } else {
                    trace!("reading fat pointer extra of type {}", ty);
                    let extra = ptr.offset(self.memory.pointer_size());
                    let extra = match self.tcx.struct_tail(ty).sty {
                        ty::TyDynamic(..) => PrimVal::Ptr(self.memory.read_ptr(extra)?),
                        ty::TySlice(..) |
                        ty::TyStr => PrimVal::from_u128(self.memory.read_usize(extra)? as u128),
                        _ => bug!("unsized primval ptr read from {:?}", ty),
                    };
                    return Ok(Some(Value::ByValPair(PrimVal::Ptr(p), extra)));
                }
            }

            ty::TyAdt(..) => {
                use rustc::ty::layout::Layout::*;
                if let CEnum { discr, signed, .. } = *self.type_layout(ty)? {
                    let size = discr.size().bytes();
                    if signed {
                        PrimVal::from_i128(self.memory.read_int(ptr, size)?)
                    } else {
                        PrimVal::from_u128(self.memory.read_uint(ptr, size)?)
                    }
                } else {
                    return Ok(None);
                }
            },

            _ => return Ok(None),
        };

        Ok(Some(Value::ByVal(val)))
    }

    pub(super) fn frame(&self) -> &Frame<'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    pub(super) fn frame_mut(&mut self) -> &mut Frame<'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    pub(super) fn mir(&self) -> MirRef<'tcx> {
        Ref::clone(&self.frame().mir)
    }

    pub(super) fn substs(&self) -> &'tcx Substs<'tcx> {
        self.frame().substs
    }

    fn unsize_into(
        &mut self,
        src: Value,
        src_ty: Ty<'tcx>,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        match (&src_ty.sty, &dest_ty.sty) {
            (&ty::TyBox(sty), &ty::TyBox(dty)) |
            (&ty::TyRef(_, ty::TypeAndMut { ty: sty, .. }), &ty::TyRef(_, ty::TypeAndMut { ty: dty, .. })) |
            (&ty::TyRef(_, ty::TypeAndMut { ty: sty, .. }), &ty::TyRawPtr(ty::TypeAndMut { ty: dty, .. })) |
            (&ty::TyRawPtr(ty::TypeAndMut { ty: sty, .. }), &ty::TyRawPtr(ty::TypeAndMut { ty: dty, .. })) => {
                // A<Struct> -> A<Trait> conversion
                let (src_pointee_ty, dest_pointee_ty) = self.tcx.struct_lockstep_tails(sty, dty);

                match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
                    (&ty::TyArray(_, length), &ty::TySlice(_)) => {
                        let ptr = src.read_ptr(&self.memory)?;
                        let len = PrimVal::from_u128(length as u128);
                        let ptr = PrimVal::Ptr(ptr);
                        self.write_value(Value::ByValPair(ptr, len), dest, dest_ty)?;
                    }
                    (&ty::TyDynamic(..), &ty::TyDynamic(..)) => {
                        // For now, upcasts are limited to changes in marker
                        // traits, and hence never actually require an actual
                        // change to the vtable.
                        self.write_value(src, dest, dest_ty)?;
                    },
                    (_, &ty::TyDynamic(ref data, _)) => {
                        let trait_ref = data.principal().unwrap().with_self_ty(self.tcx, src_pointee_ty);
                        let trait_ref = self.tcx.erase_regions(&trait_ref);
                        let vtable = self.get_vtable(trait_ref)?;
                        let ptr = src.read_ptr(&self.memory)?;
                        let ptr = PrimVal::Ptr(ptr);
                        let extra = PrimVal::Ptr(vtable);
                        self.write_value(Value::ByValPair(ptr, extra), dest, dest_ty)?;
                    },

                    _ => bug!("invalid unsizing {:?} -> {:?}", src_ty, dest_ty),
                }
            }
            (&ty::TyAdt(def_a, substs_a), &ty::TyAdt(def_b, substs_b)) => {
                // FIXME(solson)
                let dest = self.force_allocation(dest)?.to_ptr();
                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr

                assert_eq!(def_a, def_b);

                let src_fields = def_a.variants[0].fields.iter();
                let dst_fields = def_b.variants[0].fields.iter();

                //let src = adt::MaybeSizedValue::sized(src);
                //let dst = adt::MaybeSizedValue::sized(dst);
                let src_ptr = match src {
                    Value::ByRef(ptr) => ptr,
                    _ => bug!("expected pointer, got {:?}", src),
                };

                let iter = src_fields.zip(dst_fields).enumerate();
                for (i, (src_f, dst_f)) in iter {
                    let src_fty = monomorphize_field_ty(self.tcx, src_f, substs_a);
                    let dst_fty = monomorphize_field_ty(self.tcx, dst_f, substs_b);
                    if self.type_size(dst_fty)? == Some(0) {
                        continue;
                    }
                    let src_field_offset = self.get_field_offset(src_ty, i)?.bytes();
                    let dst_field_offset = self.get_field_offset(dest_ty, i)?.bytes();
                    let src_f_ptr = src_ptr.offset(src_field_offset);
                    let dst_f_ptr = dest.offset(dst_field_offset);
                    if src_fty == dst_fty {
                        self.copy(src_f_ptr, dst_f_ptr, src_fty)?;
                    } else {
                        self.unsize_into(Value::ByRef(src_f_ptr), src_fty, Lvalue::from_ptr(dst_f_ptr), dst_fty)?;
                    }
                }
            }
            _ => bug!("unsize_into: invalid conversion: {:?} -> {:?}", src_ty, dest_ty),
        }
        Ok(())
    }

    pub(super) fn dump_local(&self, lvalue: Lvalue<'tcx>) {
        let mut allocs = Vec::new();

        if let Lvalue::Local { frame, local } = lvalue {
            match self.stack[frame].get_local(local) {
                Value::ByRef(ptr) => {
                    trace!("frame[{}] {:?}:", frame, local);
                    allocs.push(ptr.alloc_id);
                }
                Value::ByVal(val) => {
                    trace!("frame[{}] {:?}: {:?}", frame, local, val);
                    if let PrimVal::Ptr(ptr) = val { allocs.push(ptr.alloc_id); }
                }
                Value::ByValPair(val1, val2) => {
                    trace!("frame[{}] {:?}: ({:?}, {:?})", frame, local, val1, val2);
                    if let PrimVal::Ptr(ptr) = val1 { allocs.push(ptr.alloc_id); }
                    if let PrimVal::Ptr(ptr) = val2 { allocs.push(ptr.alloc_id); }
                }
            }
        }

        self.memory.dump_allocs(allocs);
    }

    /// Convenience function to ensure correct usage of globals and code-sharing with locals.
    pub fn modify_global<F>(&mut self, cid: GlobalId<'tcx>, f: F) -> EvalResult<'tcx, ()>
        where F: FnOnce(&mut Self, Value) -> EvalResult<'tcx, Value>,
    {
        let mut val = *self.globals.get(&cid).expect("global not cached");
        if !val.mutable {
            return Err(EvalError::ModifiedConstantMemory);
        }
        val.value = f(self, val.value)?;
        *self.globals.get_mut(&cid).expect("already checked") = val;
        Ok(())
    }

    /// Convenience function to ensure correct usage of locals and code-sharing with globals.
    pub fn modify_local<F>(
        &mut self,
        frame: usize,
        local: mir::Local,
        f: F,
    ) -> EvalResult<'tcx, ()>
        where F: FnOnce(&mut Self, Value) -> EvalResult<'tcx, Value>,
    {
        let val = self.stack[frame].get_local(local);
        let new_val = f(self, val)?;
        self.stack[frame].set_local(local, new_val);
        // FIXME(solson): Run this when setting to Undef? (See previous version of this code.)
        // if let Value::ByRef(ptr) = self.stack[frame].get_local(local) {
        //     self.memory.deallocate(ptr)?;
        // }
        Ok(())
    }
}

impl<'tcx> Frame<'tcx> {
    pub fn get_local(&self, local: mir::Local) -> Value {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        self.locals[local.index() - 1]
    }

    fn set_local(&mut self, local: mir::Local, value: Value) {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        self.locals[local.index() - 1] = value;
    }
}

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    limits: ResourceLimits,
) {
    let mut ecx = EvalContext::new(tcx, limits);
    let mir = ecx.load_mir(def_id).expect("main function's MIR not found");

    ecx.push_stack_frame(
        def_id,
        mir.span,
        mir,
        tcx.intern_substs(&[]),
        Lvalue::from_ptr(Pointer::zst_ptr()),
        StackPopCleanup::None,
        Vec::new(),
    ).expect("could not allocate first stack frame");

    loop {
        match ecx.step() {
            Ok(true) => {}
            Ok(false) => return,
            Err(e) => {
                report(tcx, &ecx, e);
                return;
            }
        }
    }
}

fn report(tcx: TyCtxt, ecx: &EvalContext, e: EvalError) {
    let frame = ecx.stack().last().expect("stackframe was empty");
    let block = &frame.mir.basic_blocks()[frame.block];
    let span = if frame.stmt < block.statements.len() {
        block.statements[frame.stmt].source_info.span
    } else {
        block.terminator().source_info.span
    };
    let mut err = tcx.sess.struct_span_err(span, &e.to_string());
    for &Frame { def_id, substs, span, .. } in ecx.stack().iter().rev() {
        if tcx.def_key(def_id).disambiguated_data.data == DefPathData::ClosureExpr {
            err.span_note(span, "inside call to closure");
            continue;
        }
        // FIXME(solson): Find a way to do this without this Display impl hack.
        use rustc::util::ppaux;
        use std::fmt;
        struct Instance<'tcx>(DefId, &'tcx subst::Substs<'tcx>);
        impl<'tcx> ::std::panic::UnwindSafe for Instance<'tcx> {}
        impl<'tcx> ::std::panic::RefUnwindSafe for Instance<'tcx> {}
        impl<'tcx> fmt::Display for Instance<'tcx> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                ppaux::parameterized(f, self.1, self.0, &[])
            }
        }
        err.span_note(span, &format!("inside call to {}", Instance(def_id, substs)));
    }
    err.emit();
}

pub fn run_mir_passes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut passes = ::rustc::mir::transform::Passes::new();
    passes.push_hook(Box::new(::rustc_mir::transform::dump_mir::DumpMir));
    passes.push_pass(Box::new(::rustc_mir::transform::no_landing_pads::NoLandingPads));
    passes.push_pass(Box::new(::rustc_mir::transform::simplify::SimplifyCfg::new("no-landing-pads")));

    // From here on out, regions are gone.
    passes.push_pass(Box::new(::rustc_mir::transform::erase_regions::EraseRegions));

    passes.push_pass(Box::new(::rustc_mir::transform::add_call_guards::AddCallGuards));
    passes.push_pass(Box::new(::rustc_borrowck::ElaborateDrops));
    passes.push_pass(Box::new(::rustc_mir::transform::no_landing_pads::NoLandingPads));
    passes.push_pass(Box::new(::rustc_mir::transform::simplify::SimplifyCfg::new("elaborate-drops")));

    // No lifetime analysis based on borrowing can be done from here on out.
    passes.push_pass(Box::new(::rustc_mir::transform::instcombine::InstCombine::new()));
    passes.push_pass(Box::new(::rustc_mir::transform::deaggregator::Deaggregator));
    passes.push_pass(Box::new(::rustc_mir::transform::copy_prop::CopyPropagation));

    passes.push_pass(Box::new(::rustc_mir::transform::simplify::SimplifyLocals));
    passes.push_pass(Box::new(::rustc_mir::transform::add_call_guards::AddCallGuards));
    passes.push_pass(Box::new(::rustc_mir::transform::dump_mir::Marker("PreMiri")));

    passes.run_passes(tcx);
}

// TODO(solson): Upstream these methods into rustc::ty::layout.

pub(super) trait IntegerExt {
    fn size(self) -> Size;
}

impl IntegerExt for layout::Integer {
    fn size(self) -> Size {
        use rustc::ty::layout::Integer::*;
        match self {
            I1 | I8 => Size::from_bits(8),
            I16 => Size::from_bits(16),
            I32 => Size::from_bits(32),
            I64 => Size::from_bits(64),
            I128 => Size::from_bits(128),
        }
    }
}


pub fn monomorphize_field_ty<'a, 'tcx:'a >(tcx: TyCtxt<'a, 'tcx, 'tcx>, f: &ty::FieldDef, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
    let substituted = &f.ty(tcx, substs);
    tcx.normalize_associated_type(&substituted)
}

pub fn is_inhabited<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.uninhabited_from(&mut FxHashSet::default(), tcx).is_empty()
}
