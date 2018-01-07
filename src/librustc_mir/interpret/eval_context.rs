use std::collections::HashSet;
use std::fmt::Write;

use rustc::hir::def_id::DefId;
use rustc::hir::map::definitions::DefPathData;
use rustc::middle::const_val::ConstVal;
use rustc::mir;
use rustc::traits::Reveal;
use rustc::ty::layout::{self, Size, Align, HasDataLayout, LayoutOf, TyLayout};
use rustc::ty::subst::{Subst, Substs, Kind};
use rustc::ty::{self, Ty, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
use syntax::codemap::{self, DUMMY_SP};
use syntax::ast::Mutability;
use rustc::mir::interpret::{
    GlobalId, Value, Pointer, PrimVal, PrimValKind,
    EvalError, EvalResult, EvalErrorKind, MemoryPointer,
};

use super::{Place, PlaceExtra, Memory,
            HasMemory, MemoryKind, operator,
            Machine};

pub struct EvalContext<'a, 'tcx: 'a, M: Machine<'tcx>> {
    /// Stores the `Machine` instance.
    pub machine: M,

    /// The results of the type checker, from rustc.
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// Bounds in scope for polymorphic evaluations.
    pub param_env: ty::ParamEnv<'tcx>,

    /// The virtual memory system.
    pub memory: Memory<'a, 'tcx, M>,

    /// The virtual call stack.
    pub(crate) stack: Vec<Frame<'tcx>>,

    /// The maximum number of stack frames allowed
    pub(crate) stack_limit: usize,

    /// The maximum number of operations that may be executed.
    /// This prevents infinite loops and huge computations from freezing up const eval.
    /// Remove once halting problem is solved.
    pub(crate) steps_remaining: u64,
}

/// A stack frame.
pub struct Frame<'tcx> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub mir: &'tcx mir::Mir<'tcx>,

    /// The def_id and substs of the current function
    pub instance: ty::Instance<'tcx>,

    /// The span of the call site.
    pub span: codemap::Span,

    ////////////////////////////////////////////////////////////////////////////////
    // Return place and locals
    ////////////////////////////////////////////////////////////////////////////////
    /// The block to return to when returning from the current stack frame
    pub return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to.
    pub return_place: Place,

    /// The list of locals for this stack frame, stored in order as
    /// `[arguments..., variables..., temporaries...]`. The locals are stored as `Option<Value>`s.
    /// `None` represents a local that is currently dead, while a live local
    /// can either directly contain `PrimVal` or refer to some part of an `Allocation`.
    ///
    /// Before being initialized, arguments are `Value::ByVal(PrimVal::Undef)` and other locals are `None`.
    pub locals: Vec<Option<Value>>,

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
    /// isn't modifyable afterwards in case of constants.
    /// In case of `static mut`, mark the memory to ensure it's never marked as immutable through
    /// references or deallocated
    MarkStatic(Mutability),
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

#[derive(Copy, Clone, Debug)]
pub struct TyAndPacked<'tcx> {
    pub ty: Ty<'tcx>,
    pub packed: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct ValTy<'tcx> {
    pub value: Value,
    pub ty: Ty<'tcx>,
}

impl<'tcx> ::std::ops::Deref for ValTy<'tcx> {
    type Target = Value;
    fn deref(&self) -> &Value {
        &self.value
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> HasDataLayout for &'a EvalContext<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'c, 'b, 'a, 'tcx, M: Machine<'tcx>> HasDataLayout
    for &'c &'b mut EvalContext<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> layout::HasTyCtxt<'tcx> for &'a EvalContext<'a, 'tcx, M> {
    #[inline]
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> {
        self.tcx
    }
}

impl<'c, 'b, 'a, 'tcx, M: Machine<'tcx>> layout::HasTyCtxt<'tcx>
    for &'c &'b mut EvalContext<'a, 'tcx, M> {
    #[inline]
    fn tcx<'d>(&'d self) -> TyCtxt<'d, 'tcx, 'tcx> {
        self.tcx
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> LayoutOf<Ty<'tcx>> for &'a EvalContext<'a, 'tcx, M> {
    type TyLayout = EvalResult<'tcx, TyLayout<'tcx>>;

    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        (self.tcx, self.param_env).layout_of(ty)
            .map_err(|layout| EvalErrorKind::Layout(layout).into())
    }
}

impl<'c, 'b, 'a, 'tcx, M: Machine<'tcx>> LayoutOf<Ty<'tcx>>
    for &'c &'b mut EvalContext<'a, 'tcx, M> {
    type TyLayout = EvalResult<'tcx, TyLayout<'tcx>>;

    #[inline]
    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        (&**self).layout_of(ty)
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> EvalContext<'a, 'tcx, M> {
    pub fn new(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        limits: ResourceLimits,
        machine: M,
        memory_data: M::MemoryData,
    ) -> Self {
        EvalContext {
            machine,
            tcx,
            param_env,
            memory: Memory::new(tcx, limits.memory_size, memory_data),
            stack: Vec::new(),
            stack_limit: limits.stack_limit,
            steps_remaining: limits.step_limit,
        }
    }

    pub fn alloc_ptr(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, MemoryPointer> {
        let layout = self.layout_of(ty)?;
        assert!(!layout.is_unsized(), "cannot alloc memory for unsized type");

        let size = layout.size.bytes();
        self.memory.allocate(size, layout.align, Some(MemoryKind::Stack))
    }

    pub fn memory(&self) -> &Memory<'a, 'tcx, M> {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx, M> {
        &mut self.memory
    }

    pub fn stack(&self) -> &[Frame<'tcx>] {
        &self.stack
    }

    #[inline]
    pub fn cur_frame(&self) -> usize {
        assert!(self.stack.len() > 0);
        self.stack.len() - 1
    }

    pub fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        let ptr = self.memory.allocate_cached(s.as_bytes());
        Ok(Value::ByValPair(
            PrimVal::Ptr(ptr),
            PrimVal::from_u128(s.len() as u128),
        ))
    }

    pub(super) fn const_to_value(&mut self, const_val: &ConstVal<'tcx>, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        use rustc::middle::const_val::ConstVal::*;

        let primval = match *const_val {
            Integral(const_int) => PrimVal::Bytes(const_int.to_u128_unchecked()),

            Float(val) => PrimVal::Bytes(val.bits),

            Bool(b) => PrimVal::from_bool(b),
            Char(c) => PrimVal::from_char(c),

            Str(ref s) => return self.str_to_value(s),

            ByteStr(ref bs) => {
                let ptr = self.memory.allocate_cached(bs.data);
                PrimVal::Ptr(ptr)
            }

            Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                return Ok(self.read_global_as_value(GlobalId {
                    instance,
                    promoted: None,
                }, self.layout_of(ty)?));
            }

            Aggregate(..) |
            Variant(_) => bug!("should not have aggregate or variant constants in MIR"),
            // function items are zero sized and thus have no readable value
            Function(..) => PrimVal::Undef,
        };

        Ok(Value::ByVal(primval))
    }

    pub(super) fn resolve(&self, def_id: DefId, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        let substs = self.tcx.trans_apply_param_substs(self.substs(), &substs);
        ty::Instance::resolve(
            self.tcx,
            self.param_env,
            def_id,
            substs,
        ).ok_or(EvalErrorKind::TypeckError.into()) // turn error prop into a panic to expose associated type in const issue
    }

    pub(super) fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx, self.param_env, DUMMY_SP)
    }

    pub fn load_mir(
        &self,
        instance: ty::InstanceDef<'tcx>,
    ) -> EvalResult<'tcx, &'tcx mir::Mir<'tcx>> {
        // do not continue if typeck errors occurred (can only occur in local crate)
        let did = instance.def_id();
        if did.is_local() && self.tcx.has_typeck_tables(did) && self.tcx.typeck_tables_of(did).tainted_by_errors {
            return err!(TypeckError);
        }
        trace!("load mir {:?}", instance);
        match instance {
            ty::InstanceDef::Item(def_id) => {
                self.tcx.maybe_optimized_mir(def_id).ok_or_else(|| {
                    EvalErrorKind::NoMirFor(self.tcx.item_path_str(def_id)).into()
                })
            }
            _ => Ok(self.tcx.instance_mir(instance)),
        }
    }

    pub fn monomorphize(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // miri doesn't care about lifetimes, and will choke on some crazy ones
        // let's simply get rid of them
        let without_lifetimes = self.tcx.erase_regions(&ty);
        let substituted = without_lifetimes.subst(self.tcx, substs);
        let substituted = self.tcx.fully_normalize_monormophic_ty(&substituted);
        substituted
    }

    /// Return the size and aligment of the value at the given type.
    /// Note that the value does not matter if the type is sized. For unsized types,
    /// the value has to be a fat pointer, and we only care about the "extra" data in it.
    pub fn size_and_align_of_dst(
        &mut self,
        ty: Ty<'tcx>,
        value: Value,
    ) -> EvalResult<'tcx, (Size, Align)> {
        let layout = self.layout_of(ty)?;
        if !layout.is_unsized() {
            Ok(layout.size_and_align())
        } else {
            match ty.sty {
                ty::TyAdt(..) | ty::TyTuple(..) => {
                    // First get the size of all statically known fields.
                    // Don't use type_of::sizing_type_of because that expects t to be sized,
                    // and it also rounds up to alignment, which we want to avoid,
                    // as the unsized field's alignment could be smaller.
                    assert!(!ty.is_simd());
                    debug!("DST {} layout: {:?}", ty, layout);

                    let sized_size = layout.fields.offset(layout.fields.count() - 1);
                    let sized_align = layout.align;
                    debug!(
                        "DST {} statically sized prefix size: {:?} align: {:?}",
                        ty,
                        sized_size,
                        sized_align
                    );

                    // Recurse to get the size of the dynamically sized field (must be
                    // the last field).
                    let field_ty = layout.field(&self, layout.fields.count() - 1)?.ty;
                    let (unsized_size, unsized_align) =
                        self.size_and_align_of_dst(field_ty, value)?;

                    // FIXME (#26403, #27023): We should be adding padding
                    // to `sized_size` (to accommodate the `unsized_align`
                    // required of the unsized field that follows) before
                    // summing it with `sized_size`. (Note that since #26403
                    // is unfixed, we do not yet add the necessary padding
                    // here. But this is where the add would go.)

                    // Return the sum of sizes and max of aligns.
                    let size = sized_size + unsized_size;

                    // Choose max of two known alignments (combined value must
                    // be aligned according to more restrictive of the two).
                    let align = sized_align.max(unsized_align);

                    // Issue #27023: must add any necessary padding to `size`
                    // (to make it a multiple of `align`) before returning it.
                    //
                    // Namely, the returned size should be, in C notation:
                    //
                    //   `size + ((size & (align-1)) ? align : 0)`
                    //
                    // emulated via the semi-standard fast bit trick:
                    //
                    //   `(size + (align-1)) & -align`

                    Ok((size.abi_align(align), align))
                }
                ty::TyDynamic(..) => {
                    let (_, vtable) = self.into_ptr_vtable_pair(value)?;
                    // the second entry in the vtable is the dynamic size of the object.
                    self.read_size_and_align_from_vtable(vtable)
                }

                ty::TySlice(_) | ty::TyStr => {
                    let (elem_size, align) = layout.field(&self, 0)?.size_and_align();
                    let (_, len) = self.into_slice(value)?;
                    Ok((elem_size * len, align))
                }

                _ => bug!("size_of_val::<{:?}>", ty),
            }
        }
    }

    pub fn push_stack_frame(
        &mut self,
        instance: ty::Instance<'tcx>,
        span: codemap::Span,
        mir: &'tcx mir::Mir<'tcx>,
        return_place: Place,
        return_to_block: StackPopCleanup,
    ) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation += 1;

        /// Return the set of locals that have a storage annotation anywhere
        fn collect_storage_annotations<'tcx>(mir: &'tcx mir::Mir<'tcx>) -> HashSet<mir::Local> {
            use rustc::mir::StatementKind::*;

            let mut set = HashSet::new();
            for block in mir.basic_blocks() {
                for stmt in block.statements.iter() {
                    match stmt.kind {
                        StorageLive(local) |
                        StorageDead(local) => {
                            set.insert(local);
                        }
                        _ => {}
                    }
                }
            }
            set
        }

        // Subtract 1 because `local_decls` includes the ReturnMemoryPointer, but we don't store a local
        // `Value` for that.
        let num_locals = mir.local_decls.len() - 1;

        let locals = {
            let annotated_locals = collect_storage_annotations(mir);
            let mut locals = vec![None; num_locals];
            for i in 0..num_locals {
                let local = mir::Local::new(i + 1);
                if !annotated_locals.contains(&local) {
                    locals[i] = Some(Value::ByVal(PrimVal::Undef));
                }
            }
            locals
        };

        self.stack.push(Frame {
            mir,
            block: mir::START_BLOCK,
            return_to_block,
            return_place,
            locals,
            span,
            instance,
            stmt: 0,
        });

        self.memory.cur_frame = self.cur_frame();

        if self.stack.len() > self.stack_limit {
            err!(StackFrameLimitReached)
        } else {
            Ok(())
        }
    }

    pub(super) fn pop_stack_frame(&mut self) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation -= 1;
        M::end_region(self, None)?;
        let frame = self.stack.pop().expect(
            "tried to pop a stack frame, but there were none",
        );
        if !self.stack.is_empty() {
            // TODO: Is this the correct time to start considering these accesses as originating from the returned-to stack frame?
            self.memory.cur_frame = self.cur_frame();
        }
        match frame.return_to_block {
            StackPopCleanup::MarkStatic(mutable) => {
                if let Place::Ptr { ptr, .. } = frame.return_place {
                    // FIXME: to_ptr()? might be too extreme here, static zsts might reach this under certain conditions
                    self.memory.mark_static_initalized(
                        ptr.to_ptr()?.alloc_id,
                        mutable,
                    )?
                } else {
                    bug!("StackPopCleanup::MarkStatic on: {:?}", frame.return_place);
                }
            }
            StackPopCleanup::Goto(target) => self.goto_block(target),
            StackPopCleanup::None => {}
        }
        // deallocate all locals that are backed by an allocation
        for local in frame.locals {
            self.deallocate_local(local)?;
        }

        Ok(())
    }

    pub fn deallocate_local(&mut self, local: Option<Value>) -> EvalResult<'tcx> {
        if let Some(Value::ByRef(ptr, _align)) = local {
            trace!("deallocating local");
            let ptr = ptr.to_ptr()?;
            self.memory.dump_alloc(ptr.alloc_id);
            self.memory.deallocate_local(ptr)?;
        };
        Ok(())
    }

    /// Evaluate an assignment statement.
    ///
    /// There is no separate `eval_rvalue` function. Instead, the code for handling each rvalue
    /// type writes its results directly into the memory specified by the place.
    pub(super) fn eval_rvalue_into_place(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        place: &mir::Place<'tcx>,
    ) -> EvalResult<'tcx> {
        let dest = self.eval_place(place)?;
        let dest_ty = self.place_ty(place);

        use rustc::mir::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let value = self.eval_operand(operand)?.value;
                let valty = ValTy {
                    value,
                    ty: dest_ty,
                };
                self.write_value(valty, dest)?;
            }

            BinaryOp(bin_op, ref left, ref right) => {
                let left = self.eval_operand(left)?;
                let right = self.eval_operand(right)?;
                if self.intrinsic_overflowing(
                    bin_op,
                    left,
                    right,
                    dest,
                    dest_ty,
                )?
                {
                    // There was an overflow in an unchecked binop.  Right now, we consider this an error and bail out.
                    // The rationale is that the reason rustc emits unchecked binops in release mode (vs. the checked binops
                    // it emits in debug mode) is performance, but it doesn't cost us any performance in miri.
                    // If, however, the compiler ever starts transforming unchecked intrinsics into unchecked binops,
                    // we have to go back to just ignoring the overflow here.
                    return err!(OverflowingMath);
                }
            }

            CheckedBinaryOp(bin_op, ref left, ref right) => {
                let left = self.eval_operand(left)?;
                let right = self.eval_operand(right)?;
                self.intrinsic_with_overflow(
                    bin_op,
                    left,
                    right,
                    dest,
                    dest_ty,
                )?;
            }

            UnaryOp(un_op, ref operand) => {
                let val = self.eval_operand_to_primval(operand)?;
                let kind = self.ty_to_primval_kind(dest_ty)?;
                self.write_primval(
                    dest,
                    operator::unary_op(un_op, val, kind)?,
                    dest_ty,
                )?;
            }

            Aggregate(ref kind, ref operands) => {
                self.inc_step_counter_and_check_limit(operands.len() as u64)?;

                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_def, variant_index, _, active_field_index) => {
                        self.write_discriminant_value(dest_ty, dest, variant_index)?;
                        if adt_def.is_enum() {
                            (self.place_downcast(dest, variant_index)?, active_field_index)
                        } else {
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None)
                };

                let layout = self.layout_of(dest_ty)?;
                for (i, operand) in operands.iter().enumerate() {
                    let value = self.eval_operand(operand)?;
                    // Ignore zero-sized fields.
                    if !self.layout_of(value.ty)?.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        let (field_dest, _) = self.place_field(dest, mir::Field::new(field_index), layout)?;
                        self.write_value(value, field_dest)?;
                    }
                }
            }

            Repeat(ref operand, _) => {
                let (elem_ty, length) = match dest_ty.sty {
                    ty::TyArray(elem_ty, n) => (elem_ty, n.val.to_const_int().unwrap().to_u64().unwrap()),
                    _ => {
                        bug!(
                            "tried to assign array-repeat to non-array type {:?}",
                            dest_ty
                        )
                    }
                };
                let elem_size = self.layout_of(elem_ty)?.size.bytes();
                let value = self.eval_operand(operand)?.value;

                let (dest, dest_align) = self.force_allocation(dest)?.to_ptr_align();

                // FIXME: speed up repeat filling
                for i in 0..length {
                    let elem_dest = dest.offset(i * elem_size, &self)?;
                    self.write_value_to_ptr(value, elem_dest, dest_align, elem_ty)?;
                }
            }

            Len(ref place) => {
                // FIXME(CTFE): don't allow computing the length of arrays in const eval
                let src = self.eval_place(place)?;
                let ty = self.place_ty(place);
                let (_, len) = src.elem_ty_and_len(ty);
                self.write_primval(
                    dest,
                    PrimVal::from_u128(len as u128),
                    dest_ty,
                )?;
            }

            Ref(_, _, ref place) => {
                let src = self.eval_place(place)?;
                // We ignore the alignment of the place here -- special handling for packed structs ends
                // at the `&` operator.
                let (ptr, _align, extra) = self.force_allocation(src)?.to_ptr_align_extra();

                let val = match extra {
                    PlaceExtra::None => ptr.to_value(),
                    PlaceExtra::Length(len) => ptr.to_value_with_len(len),
                    PlaceExtra::Vtable(vtable) => ptr.to_value_with_vtable(vtable),
                    PlaceExtra::DowncastVariant(..) => {
                        bug!("attempted to take a reference to an enum downcast place")
                    }
                };
                let valty = ValTy {
                    value: val,
                    ty: dest_ty,
                };
                self.write_value(valty, dest)?;
            }

            NullaryOp(mir::NullOp::Box, ty) => {
                let ty = self.monomorphize(ty, self.substs());
                M::box_alloc(self, ty, dest)?;
            }

            NullaryOp(mir::NullOp::SizeOf, ty) => {
                let ty = self.monomorphize(ty, self.substs());
                let layout = self.layout_of(ty)?;
                assert!(!layout.is_unsized(),
                        "SizeOf nullary MIR operator called for unsized type");
                self.write_primval(
                    dest,
                    PrimVal::from_u128(layout.size.bytes() as u128),
                    dest_ty,
                )?;
            }

            Cast(kind, ref operand, cast_ty) => {
                debug_assert_eq!(self.monomorphize(cast_ty, self.substs()), dest_ty);
                use rustc::mir::CastKind::*;
                match kind {
                    Unsize => {
                        let src = self.eval_operand(operand)?;
                        let src_layout = self.layout_of(src.ty)?;
                        let dst_layout = self.layout_of(dest_ty)?;
                        self.unsize_into(src.value, src_layout, dest, dst_layout)?;
                    }

                    Misc => {
                        let src = self.eval_operand(operand)?;
                        if self.type_is_fat_ptr(src.ty) {
                            match (src.value, self.type_is_fat_ptr(dest_ty)) {
                                (Value::ByRef { .. }, _) |
                                (Value::ByValPair(..), true) => {
                                    let valty = ValTy {
                                        value: src.value,
                                        ty: dest_ty,
                                    };
                                    self.write_value(valty, dest)?;
                                }
                                (Value::ByValPair(data, _), false) => {
                                    let valty = ValTy {
                                        value: Value::ByVal(data),
                                        ty: dest_ty,
                                    };
                                    self.write_value(valty, dest)?;
                                }
                                (Value::ByVal(_), _) => bug!("expected fat ptr"),
                            }
                        } else {
                            let src_val = self.value_to_primval(src)?;
                            let dest_val = self.cast_primval(src_val, src.ty, dest_ty)?;
                            let valty = ValTy {
                                value: Value::ByVal(dest_val),
                                ty: dest_ty,
                            };
                            self.write_value(valty, dest)?;
                        }
                    }

                    ReifyFnPointer => {
                        match self.eval_operand(operand)?.ty.sty {
                            ty::TyFnDef(def_id, substs) => {
                                let instance = self.resolve(def_id, substs)?;
                                let fn_ptr = self.memory.create_fn_alloc(instance);
                                let valty = ValTy {
                                    value: Value::ByVal(PrimVal::Ptr(fn_ptr)),
                                    ty: dest_ty,
                                };
                                self.write_value(valty, dest)?;
                            }
                            ref other => bug!("reify fn pointer on {:?}", other),
                        }
                    }

                    UnsafeFnPointer => {
                        match dest_ty.sty {
                            ty::TyFnPtr(_) => {
                                let mut src = self.eval_operand(operand)?;
                                src.ty = dest_ty;
                                self.write_value(src, dest)?;
                            }
                            ref other => bug!("fn to unsafe fn cast on {:?}", other),
                        }
                    }

                    ClosureFnPointer => {
                        match self.eval_operand(operand)?.ty.sty {
                            ty::TyClosure(def_id, substs) => {
                                let substs = self.tcx.trans_apply_param_substs(self.substs(), &substs);
                                let instance = ty::Instance::resolve_closure(
                                    self.tcx,
                                    def_id,
                                    substs,
                                    ty::ClosureKind::FnOnce,
                                );
                                let fn_ptr = self.memory.create_fn_alloc(instance);
                                let valty = ValTy {
                                    value: Value::ByVal(PrimVal::Ptr(fn_ptr)),
                                    ty: dest_ty,
                                };
                                self.write_value(valty, dest)?;
                            }
                            ref other => bug!("closure fn pointer on {:?}", other),
                        }
                    }
                }
            }

            Discriminant(ref place) => {
                let ty = self.place_ty(place);
                let place = self.eval_place(place)?;
                let discr_val = self.read_discriminant_value(place, ty)?;
                if let ty::TyAdt(adt_def, _) = ty.sty {
                    trace!("Read discriminant {}, valid discriminants {:?}", discr_val, adt_def.discriminants(self.tcx).collect::<Vec<_>>());
                    if adt_def.discriminants(self.tcx).all(|v| {
                        discr_val != v.to_u128_unchecked()
                    })
                    {
                        return err!(InvalidDiscriminant);
                    }
                    self.write_primval(dest, PrimVal::Bytes(discr_val), dest_ty)?;
                } else {
                    bug!("rustc only generates Rvalue::Discriminant for enums");
                }
            }
        }

        if log_enabled!(::log::Level::Trace) {
            self.dump_local(dest);
        }

        Ok(())
    }

    pub(super) fn type_is_fat_ptr(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyRawPtr(ref tam) |
            ty::TyRef(_, ref tam) => !self.type_is_sized(tam.ty),
            ty::TyAdt(def, _) if def.is_box() => !self.type_is_sized(ty.boxed_ty()),
            _ => false,
        }
    }

    pub(super) fn eval_operand_to_primval(
        &mut self,
        op: &mir::Operand<'tcx>,
    ) -> EvalResult<'tcx, PrimVal> {
        let valty = self.eval_operand(op)?;
        self.value_to_primval(valty)
    }

    pub(crate) fn operands_to_args(
        &mut self,
        ops: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx, Vec<ValTy<'tcx>>> {
        ops.into_iter()
            .map(|op| self.eval_operand(op))
            .collect()
    }

    pub fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, ValTy<'tcx>> {
        use rustc::mir::Operand::*;
        let ty = self.monomorphize(op.ty(self.mir(), self.tcx), self.substs());
        match *op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(ref place) |
            Move(ref place) => {
                Ok(ValTy {
                    value: self.eval_and_read_place(place)?,
                    ty
                })
            },

            Constant(ref constant) => {
                use rustc::mir::Literal;
                let mir::Constant { ref literal, .. } = **constant;
                let value = match *literal {
                    Literal::Value { ref value } => self.const_to_value(&value.val, ty)?,

                    Literal::Promoted { index } => {
                        self.read_global_as_value(GlobalId {
                            instance: self.frame().instance,
                            promoted: Some(index),
                        }, self.layout_of(ty)?)
                    }
                };

                Ok(ValTy {
                    value,
                    ty,
                })
            }
        }
    }

    pub fn read_discriminant_value(
        &mut self,
        place: Place,
        ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, u128> {
        let layout = self.layout_of(ty)?;
        //trace!("read_discriminant_value {:#?}", layout);

        match layout.variants {
            layout::Variants::Single { index } => {
                return Ok(index as u128);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }

        let (discr_place, discr) = self.place_field(place, mir::Field::new(0), layout)?;
        let raw_discr = self.value_to_primval(ValTy {
            value: self.read_place(discr_place)?,
            ty: discr.ty
        })?;
        let discr_val = match layout.variants {
            layout::Variants::Single { .. } => bug!(),
            layout::Variants::Tagged { .. } => raw_discr.to_bytes()?,
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let variants_start = niche_variants.start as u128;
                let variants_end = niche_variants.end as u128;
                match raw_discr {
                    PrimVal::Ptr(_) => {
                        assert!(niche_start == 0);
                        assert!(variants_start == variants_end);
                        dataful_variant as u128
                    },
                    PrimVal::Bytes(raw_discr) => {
                        let discr = raw_discr.wrapping_sub(niche_start)
                            .wrapping_add(variants_start);
                        if variants_start <= discr && discr <= variants_end {
                            discr
                        } else {
                            dataful_variant as u128
                        }
                    },
                    PrimVal::Undef => return err!(ReadUndefBytes),
                }
            }
        };

        Ok(discr_val)
    }


    pub(crate) fn write_discriminant_value(
        &mut self,
        dest_ty: Ty<'tcx>,
        dest: Place,
        variant_index: usize,
    ) -> EvalResult<'tcx> {
        let layout = self.layout_of(dest_ty)?;

        match layout.variants {
            layout::Variants::Single { index } => {
                if index != variant_index {
                    // If the layout of an enum is `Single`, all
                    // other variants are necessarily uninhabited.
                    assert_eq!(layout.for_variant(&self, variant_index).abi,
                               layout::Abi::Uninhabited);
                }
            }
            layout::Variants::Tagged { .. } => {
                let discr_val = dest_ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(self.tcx, variant_index)
                    .to_u128_unchecked();

                let (discr_dest, discr) = self.place_field(dest, mir::Field::new(0), layout)?;
                self.write_primval(discr_dest, PrimVal::Bytes(discr_val), discr.ty)?;
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                if variant_index != dataful_variant {
                    let (niche_dest, niche) =
                        self.place_field(dest, mir::Field::new(0), layout)?;
                    let niche_value = ((variant_index - niche_variants.start) as u128)
                        .wrapping_add(niche_start);
                    self.write_primval(niche_dest, PrimVal::Bytes(niche_value), niche.ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn read_global_as_value(&self, gid: GlobalId, layout: TyLayout) -> Value {
        let alloc = self.tcx.interpret_interner.borrow().get_cached(gid).expect("global not cached");
        Value::ByRef(MemoryPointer::new(alloc, 0).into(), layout.align)
    }

    pub fn force_allocation(&mut self, place: Place) -> EvalResult<'tcx, Place> {
        let new_place = match place {
            Place::Local { frame, local } => {
                // -1 since we don't store the return value
                match self.stack[frame].locals[local.index() - 1] {
                    None => return err!(DeadLocal),
                    Some(Value::ByRef(ptr, align)) => {
                        Place::Ptr {
                            ptr,
                            align,
                            extra: PlaceExtra::None,
                        }
                    }
                    Some(val) => {
                        let ty = self.stack[frame].mir.local_decls[local].ty;
                        let ty = self.monomorphize(ty, self.stack[frame].instance.substs);
                        let layout = self.layout_of(ty)?;
                        let ptr = self.alloc_ptr(ty)?;
                        self.stack[frame].locals[local.index() - 1] =
                            Some(Value::ByRef(ptr.into(), layout.align)); // it stays live
                        let place = Place::from_ptr(ptr, layout.align);
                        self.write_value(ValTy { value: val, ty }, place)?;
                        place
                    }
                }
            }
            Place::Ptr { .. } => place,
        };
        Ok(new_place)
    }

    /// ensures this Value is not a ByRef
    pub fn follow_by_ref_value(
        &self,
        value: Value,
        ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        match value {
            Value::ByRef(ptr, align) => {
                self.read_value(ptr, align, ty)
            }
            other => Ok(other),
        }
    }

    pub fn value_to_primval(
        &self,
        ValTy { value, ty } : ValTy<'tcx>,
    ) -> EvalResult<'tcx, PrimVal> {
        match self.follow_by_ref_value(value, ty)? {
            Value::ByRef { .. } => bug!("follow_by_ref_value can't result in `ByRef`"),

            Value::ByVal(primval) => {
                // TODO: Do we really want insta-UB here?
                self.ensure_valid_value(primval, ty)?;
                Ok(primval)
            }

            Value::ByValPair(..) => bug!("value_to_primval can't work with fat pointers"),
        }
    }

    pub fn write_ptr(&mut self, dest: Place, val: Pointer, dest_ty: Ty<'tcx>) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: val.to_value(),
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    pub fn write_primval(
        &mut self,
        dest: Place,
        val: PrimVal,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: Value::ByVal(val),
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    pub fn write_value(
        &mut self,
        ValTy { value: src_val, ty: dest_ty } : ValTy<'tcx>,
        dest: Place,
    ) -> EvalResult<'tcx> {
        //trace!("Writing {:?} to {:?} at type {:?}", src_val, dest, dest_ty);
        // Note that it is really important that the type here is the right one, and matches the type things are read at.
        // In case `src_val` is a `ByValPair`, we don't do any magic here to handle padding properly, which is only
        // correct if we never look at this data with the wrong type.

        match dest {
            Place::Ptr { ptr, align, extra } => {
                assert_eq!(extra, PlaceExtra::None);
                self.write_value_to_ptr(src_val, ptr, align, dest_ty)
            }

            Place::Local { frame, local } => {
                let dest = self.stack[frame].get_local(local)?;
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
    fn write_value_possibly_by_val<F: FnOnce(&mut Self, Value) -> EvalResult<'tcx>>(
        &mut self,
        src_val: Value,
        write_dest: F,
        old_dest_val: Value,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        if let Value::ByRef(dest_ptr, align) = old_dest_val {
            // If the value is already `ByRef` (that is, backed by an `Allocation`),
            // then we must write the new value into this allocation, because there may be
            // other pointers into the allocation. These other pointers are logically
            // pointers into the local variable, and must be able to observe the change.
            //
            // Thus, it would be an error to replace the `ByRef` with a `ByVal`, unless we
            // knew for certain that there were no outstanding pointers to this allocation.
            self.write_value_to_ptr(src_val, dest_ptr, align, dest_ty)?;
        } else if let Value::ByRef(src_ptr, align) = src_val {
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
            if let Ok(Some(src_val)) = self.try_read_value(src_ptr, align, dest_ty) {
                write_dest(self, src_val)?;
            } else {
                let dest_ptr = self.alloc_ptr(dest_ty)?.into();
                let layout = self.layout_of(dest_ty)?;
                self.memory.copy(src_ptr, align.min(layout.align), dest_ptr, layout.align, layout.size.bytes(), false)?;
                write_dest(self, Value::ByRef(dest_ptr, layout.align))?;
            }
        } else {
            // Finally, we have the simple case where neither source nor destination are
            // `ByRef`. We may simply copy the source value over the the destintion.
            write_dest(self, src_val)?;
        }
        Ok(())
    }

    pub fn write_value_to_ptr(
        &mut self,
        value: Value,
        dest: Pointer,
        dest_align: Align,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        trace!("write_value_to_ptr: {:#?}", value);
        let layout = self.layout_of(dest_ty)?;
        match value {
            Value::ByRef(ptr, align) => {
                self.memory.copy(ptr, align.min(layout.align), dest, dest_align.min(layout.align), layout.size.bytes(), false)
            }
            Value::ByVal(primval) => {
                match layout.abi {
                    layout::Abi::Scalar(_) => {}
                    _ if primval.is_undef() => {}
                    _ => bug!("write_value_to_ptr: invalid ByVal layout: {:#?}", layout)
                }
                // TODO: Do we need signedness?
                self.memory.write_primval(dest.to_ptr()?, dest_align, primval, layout.size.bytes(), false)
            }
            Value::ByValPair(a_val, b_val) => {
                let ptr = dest.to_ptr()?;
                trace!("write_value_to_ptr valpair: {:#?}", layout);
                let (a, b) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (&a.value, &b.value),
                    _ => bug!("write_value_to_ptr: invalid ByValPair layout: {:#?}", layout)
                };
                let (a_size, b_size) = (a.size(&self), b.size(&self));
                let a_ptr = ptr;
                let b_offset = a_size.abi_align(b.align(&self));
                let b_ptr = ptr.offset(b_offset.bytes(), &self)?.into();
                // TODO: What about signedess?
                self.memory.write_primval(a_ptr, dest_align, a_val, a_size.bytes(), false)?;
                self.memory.write_primval(b_ptr, dest_align, b_val, b_size.bytes(), false)
            }
        }
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
                    Isize => self.memory.pointer_size(),
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
                    Usize => self.memory.pointer_size(),
                };
                PrimValKind::from_uint_size(size)
            }

            ty::TyFloat(FloatTy::F32) => PrimValKind::F32,
            ty::TyFloat(FloatTy::F64) => PrimValKind::F64,

            ty::TyFnPtr(_) => PrimValKind::FnPtr,

            ty::TyRef(_, ref tam) |
            ty::TyRawPtr(ref tam) if self.type_is_sized(tam.ty) => PrimValKind::Ptr,

            ty::TyAdt(def, _) if def.is_box() => PrimValKind::Ptr,

            ty::TyAdt(..) => {
                match self.layout_of(ty)?.abi {
                    layout::Abi::Scalar(ref scalar) => {
                        use rustc::ty::layout::Primitive::*;
                        match scalar.value {
                            Int(i, false) => PrimValKind::from_uint_size(i.size().bytes()),
                            Int(i, true) => PrimValKind::from_int_size(i.size().bytes()),
                            F32 => PrimValKind::F32,
                            F64 => PrimValKind::F64,
                            Pointer => PrimValKind::Ptr,
                        }
                    }

                    _ => return err!(TypeNotPrimitive(ty)),
                }
            }

            _ => return err!(TypeNotPrimitive(ty)),
        };

        Ok(kind)
    }

    fn ensure_valid_value(&self, val: PrimVal, ty: Ty<'tcx>) -> EvalResult<'tcx> {
        match ty.sty {
            ty::TyBool if val.to_bytes()? > 1 => err!(InvalidBool),

            ty::TyChar if ::std::char::from_u32(val.to_bytes()? as u32).is_none() => {
                err!(InvalidChar(val.to_bytes()? as u32 as u128))
            }

            _ => Ok(()),
        }
    }

    pub fn read_value(&self, ptr: Pointer, align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, align, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    pub(crate) fn read_ptr(
        &self,
        ptr: MemoryPointer,
        ptr_align: Align,
        pointee_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        let ptr_size = self.memory.pointer_size();
        let p: Pointer = self.memory.read_ptr_sized_unsigned(ptr, ptr_align)?.into();
        if self.type_is_sized(pointee_ty) {
            Ok(p.to_value())
        } else {
            trace!("reading fat pointer extra of type {}", pointee_ty);
            let extra = ptr.offset(ptr_size, self)?;
            match self.tcx.struct_tail(pointee_ty).sty {
                ty::TyDynamic(..) => Ok(p.to_value_with_vtable(
                    self.memory.read_ptr_sized_unsigned(extra, ptr_align)?.to_ptr()?,
                )),
                ty::TySlice(..) | ty::TyStr => Ok(
                    p.to_value_with_len(self.memory.read_ptr_sized_unsigned(extra, ptr_align)?.to_bytes()? as u64),
                ),
                _ => bug!("unsized primval ptr read from {:?}", pointee_ty),
            }
        }
    }

    pub fn try_read_value(&self, ptr: Pointer, ptr_align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        use syntax::ast::FloatTy;

        let ptr = ptr.to_ptr()?;
        let val = match ty.sty {
            ty::TyBool => {
                let val = self.memory.read_primval(ptr, ptr_align, 1, false)?;
                let val = match val {
                    PrimVal::Bytes(0) => false,
                    PrimVal::Bytes(1) => true,
                    // TODO: This seems a little overeager, should reading at bool type already be insta-UB?
                    _ => return err!(InvalidBool),
                };
                PrimVal::from_bool(val)
            }
            ty::TyChar => {
                let c = self.memory.read_primval(ptr, ptr_align, 4, false)?.to_bytes()? as u32;
                match ::std::char::from_u32(c) {
                    Some(ch) => PrimVal::from_char(ch),
                    None => return err!(InvalidChar(c as u128)),
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
                    Isize => self.memory.pointer_size(),
                };
                self.memory.read_primval(ptr, ptr_align, size, true)?
            }

            ty::TyUint(uint_ty) => {
                use syntax::ast::UintTy::*;
                let size = match uint_ty {
                    U8 => 1,
                    U16 => 2,
                    U32 => 4,
                    U64 => 8,
                    U128 => 16,
                    Usize => self.memory.pointer_size(),
                };
                self.memory.read_primval(ptr, ptr_align, size, false)?
            }

            ty::TyFloat(FloatTy::F32) => {
                PrimVal::Bytes(self.memory.read_primval(ptr, ptr_align, 4, false)?.to_bytes()?)
            }
            ty::TyFloat(FloatTy::F64) => {
                PrimVal::Bytes(self.memory.read_primval(ptr, ptr_align, 8, false)?.to_bytes()?)
            }

            ty::TyFnPtr(_) => self.memory.read_ptr_sized_unsigned(ptr, ptr_align)?,
            ty::TyRef(_, ref tam) |
            ty::TyRawPtr(ref tam) => return self.read_ptr(ptr, ptr_align, tam.ty).map(Some),

            ty::TyAdt(def, _) => {
                if def.is_box() {
                    return self.read_ptr(ptr, ptr_align, ty.boxed_ty()).map(Some);
                }

                if let layout::Abi::Scalar(ref scalar) = self.layout_of(ty)?.abi {
                    let mut signed = false;
                    if let layout::Int(_, s) = scalar.value {
                        signed = s;
                    }
                    let size = scalar.value.size(self).bytes();
                    self.memory.read_primval(ptr, ptr_align, size, signed)?
                } else {
                    return Ok(None);
                }
            }

            _ => return Ok(None),
        };

        Ok(Some(Value::ByVal(val)))
    }

    pub fn frame(&self) -> &Frame<'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    pub fn frame_mut(&mut self) -> &mut Frame<'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    pub(super) fn mir(&self) -> &'tcx mir::Mir<'tcx> {
        self.frame().mir
    }

    pub fn substs(&self) -> &'tcx Substs<'tcx> {
        if let Some(frame) = self.stack.last() {
            frame.instance.substs
        } else {
            Substs::empty()
        }
    }

    fn unsize_into_ptr(
        &mut self,
        src: Value,
        src_ty: Ty<'tcx>,
        dest: Place,
        dest_ty: Ty<'tcx>,
        sty: Ty<'tcx>,
        dty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        // A<Struct> -> A<Trait> conversion
        let (src_pointee_ty, dest_pointee_ty) = self.tcx.struct_lockstep_tails(sty, dty);

        match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
            (&ty::TyArray(_, length), &ty::TySlice(_)) => {
                let ptr = self.into_ptr(src)?;
                // u64 cast is from usize to u64, which is always good
                let valty = ValTy {
                    value: ptr.to_value_with_len(length.val.to_const_int().unwrap().to_u64().unwrap() ),
                    ty: dest_ty,
                };
                self.write_value(valty, dest)
            }
            (&ty::TyDynamic(..), &ty::TyDynamic(..)) => {
                // For now, upcasts are limited to changes in marker
                // traits, and hence never actually require an actual
                // change to the vtable.
                let valty = ValTy {
                    value: src,
                    ty: dest_ty,
                };
                self.write_value(valty, dest)
            }
            (_, &ty::TyDynamic(ref data, _)) => {
                let trait_ref = data.principal().unwrap().with_self_ty(
                    self.tcx,
                    src_pointee_ty,
                );
                let trait_ref = self.tcx.erase_regions(&trait_ref);
                let vtable = self.get_vtable(src_pointee_ty, trait_ref)?;
                let ptr = self.into_ptr(src)?;
                let valty = ValTy {
                    value: ptr.to_value_with_vtable(vtable),
                    ty: dest_ty,
                };
                self.write_value(valty, dest)
            }

            _ => bug!("invalid unsizing {:?} -> {:?}", src_ty, dest_ty),
        }
    }

    fn unsize_into(
        &mut self,
        src: Value,
        src_layout: TyLayout<'tcx>,
        dst: Place,
        dst_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx> {
        match (&src_layout.ty.sty, &dst_layout.ty.sty) {
            (&ty::TyRef(_, ref s), &ty::TyRef(_, ref d)) |
            (&ty::TyRef(_, ref s), &ty::TyRawPtr(ref d)) |
            (&ty::TyRawPtr(ref s), &ty::TyRawPtr(ref d)) => {
                self.unsize_into_ptr(src, src_layout.ty, dst, dst_layout.ty, s.ty, d.ty)
            }
            (&ty::TyAdt(def_a, _), &ty::TyAdt(def_b, _)) => {
                assert_eq!(def_a, def_b);
                if def_a.is_box() || def_b.is_box() {
                    if !def_a.is_box() || !def_b.is_box() {
                        bug!("invalid unsizing between {:?} -> {:?}", src_layout, dst_layout);
                    }
                    return self.unsize_into_ptr(
                        src,
                        src_layout.ty,
                        dst,
                        dst_layout.ty,
                        src_layout.ty.boxed_ty(),
                        dst_layout.ty.boxed_ty(),
                    );
                }

                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr
                for i in 0..src_layout.fields.count() {
                    let (dst_f_place, dst_field) =
                        self.place_field(dst, mir::Field::new(i), dst_layout)?;
                    if dst_field.is_zst() {
                        continue;
                    }
                    let (src_f_value, src_field) = match src {
                        Value::ByRef(ptr, align) => {
                            let src_place = Place::from_primval_ptr(ptr, align);
                            let (src_f_place, src_field) =
                                self.place_field(src_place, mir::Field::new(i), src_layout)?;
                            (self.read_place(src_f_place)?, src_field)
                        }
                        Value::ByVal(_) | Value::ByValPair(..) => {
                            let src_field = src_layout.field(&self, i)?;
                            assert_eq!(src_layout.fields.offset(i).bytes(), 0);
                            assert_eq!(src_field.size, src_layout.size);
                            (src, src_field)
                        }
                    };
                    if src_field.ty == dst_field.ty {
                        self.write_value(ValTy {
                            value: src_f_value,
                            ty: src_field.ty,
                        }, dst_f_place)?;
                    } else {
                        self.unsize_into(src_f_value, src_field, dst_f_place, dst_field)?;
                    }
                }
                Ok(())
            }
            _ => {
                bug!(
                    "unsize_into: invalid conversion: {:?} -> {:?}",
                    src_layout,
                    dst_layout
                )
            }
        }
    }

    pub fn dump_local(&self, place: Place) {
        // Debug output
        match place {
            Place::Local { frame, local } => {
                let mut allocs = Vec::new();
                let mut msg = format!("{:?}", local);
                if frame != self.cur_frame() {
                    write!(msg, " ({} frames up)", self.cur_frame() - frame).unwrap();
                }
                write!(msg, ":").unwrap();

                match self.stack[frame].get_local(local) {
                    Err(EvalError { kind: EvalErrorKind::DeadLocal, .. }) => {
                        write!(msg, " is dead").unwrap();
                    }
                    Err(err) => {
                        panic!("Failed to access local: {:?}", err);
                    }
                    Ok(Value::ByRef(ptr, align)) => {
                        match ptr.into_inner_primval() {
                            PrimVal::Ptr(ptr) => {
                                write!(msg, " by align({}) ref:", align.abi()).unwrap();
                                allocs.push(ptr.alloc_id);
                            }
                            ptr => write!(msg, " integral by ref: {:?}", ptr).unwrap(),
                        }
                    }
                    Ok(Value::ByVal(val)) => {
                        write!(msg, " {:?}", val).unwrap();
                        if let PrimVal::Ptr(ptr) = val {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                    Ok(Value::ByValPair(val1, val2)) => {
                        write!(msg, " ({:?}, {:?})", val1, val2).unwrap();
                        if let PrimVal::Ptr(ptr) = val1 {
                            allocs.push(ptr.alloc_id);
                        }
                        if let PrimVal::Ptr(ptr) = val2 {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                }

                trace!("{}", msg);
                self.memory.dump_allocs(allocs);
            }
            Place::Ptr { ptr, align, .. } => {
                match ptr.into_inner_primval() {
                    PrimVal::Ptr(ptr) => {
                        trace!("by align({}) ref:", align.abi());
                        self.memory.dump_alloc(ptr.alloc_id);
                    }
                    ptr => trace!(" integral by ref: {:?}", ptr),
                }
            }
        }
    }

    /// Convenience function to ensure correct usage of locals
    pub fn modify_local<F>(&mut self, frame: usize, local: mir::Local, f: F) -> EvalResult<'tcx>
    where
        F: FnOnce(&mut Self, Value) -> EvalResult<'tcx, Value>,
    {
        let val = self.stack[frame].get_local(local)?;
        let new_val = f(self, val)?;
        self.stack[frame].set_local(local, new_val)?;
        // FIXME(solson): Run this when setting to Undef? (See previous version of this code.)
        // if let Value::ByRef(ptr) = self.stack[frame].get_local(local) {
        //     self.memory.deallocate(ptr)?;
        // }
        Ok(())
    }

    pub fn report(&self, e: &mut EvalError) {
        if let Some(ref mut backtrace) = e.backtrace {
            let mut trace_text = "\n\nAn error occurred in miri:\n".to_string();
            backtrace.resolve();
            write!(trace_text, "backtrace frames: {}\n", backtrace.frames().len()).unwrap();
            'frames: for (i, frame) in backtrace.frames().iter().enumerate() {
                if frame.symbols().is_empty() {
                    write!(trace_text, "{}: no symbols\n", i).unwrap();
                }
                for symbol in frame.symbols() {
                    write!(trace_text, "{}: ", i).unwrap();
                    if let Some(name) = symbol.name() {
                        write!(trace_text, "{}\n", name).unwrap();
                    } else {
                        write!(trace_text, "<unknown>\n").unwrap();
                    }
                    write!(trace_text, "\tat ").unwrap();
                    if let Some(file_path) = symbol.filename() {
                        write!(trace_text, "{}", file_path.display()).unwrap();
                    } else {
                        write!(trace_text, "<unknown_file>").unwrap();
                    }
                    if let Some(line) = symbol.lineno() {
                        write!(trace_text, ":{}\n", line).unwrap();
                    } else {
                        write!(trace_text, "\n").unwrap();
                    }
                }
            }
            error!("{}", trace_text);
        }
        if let Some(frame) = self.stack().last() {
            let block = &frame.mir.basic_blocks()[frame.block];
            let span = if frame.stmt < block.statements.len() {
                block.statements[frame.stmt].source_info.span
            } else {
                block.terminator().source_info.span
            };
            let mut err = self.tcx.sess.struct_span_err(span, &e.to_string());
            for &Frame { instance, span, .. } in self.stack().iter().rev() {
                if self.tcx.def_key(instance.def_id()).disambiguated_data.data ==
                    DefPathData::ClosureExpr
                {
                    err.span_note(span, "inside call to closure");
                    continue;
                }
                err.span_note(span, &format!("inside call to {}", instance));
            }
            err.emit();
        } else {
            self.tcx.sess.err(&e.to_string());
        }
    }
}

impl<'tcx> Frame<'tcx> {
    pub fn get_local(&self, local: mir::Local) -> EvalResult<'tcx, Value> {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        self.locals[local.index() - 1].ok_or(EvalErrorKind::DeadLocal.into())
    }

    fn set_local(&mut self, local: mir::Local, value: Value) -> EvalResult<'tcx> {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        match self.locals[local.index() - 1] {
            None => err!(DeadLocal),
            Some(ref mut local) => {
                *local = value;
                Ok(())
            }
        }
    }

    pub fn storage_live(&mut self, local: mir::Local) -> EvalResult<'tcx, Option<Value>> {
        trace!("{:?} is now live", local);

        let old = self.locals[local.index() - 1];
        self.locals[local.index() - 1] = Some(Value::ByVal(PrimVal::Undef)); // StorageLive *always* kills the value that's currently stored
        return Ok(old);
    }

    /// Returns the old value of the local
    pub fn storage_dead(&mut self, local: mir::Local) -> EvalResult<'tcx, Option<Value>> {
        trace!("{:?} is now dead", local);

        let old = self.locals[local.index() - 1];
        self.locals[local.index() - 1] = None;
        return Ok(old);
    }
}

// TODO(solson): Upstream these methods into rustc::ty::layout.

pub fn resolve_drop_in_place<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: Ty<'tcx>,
) -> ty::Instance<'tcx> {
    let def_id = tcx.require_lang_item(::rustc::middle::lang_items::DropInPlaceFnLangItem);
    let substs = tcx.intern_substs(&[Kind::from(ty)]);
    ty::Instance::resolve(tcx, ty::ParamEnv::empty(Reveal::All), def_id, substs).unwrap()
}
