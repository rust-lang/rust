use std::fmt::Write;

use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::hir::map::definitions::DefPathData;
use rustc::middle::const_val::ConstVal;
use rustc::mir;
use rustc::ty::layout::{self, Size, Align, HasDataLayout, IntegerExt, LayoutOf, TyLayout};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, TypeAndMut};
use rustc::ty::query::TyCtxtAt;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::middle::const_val::FrameInfo;
use syntax::codemap::{self, Span};
use syntax::ast::Mutability;
use rustc::mir::interpret::{
    GlobalId, Value, Scalar,
    EvalResult, EvalErrorKind, Pointer, ConstValue,
};
use std::mem;

use super::{Place, PlaceExtra, Memory,
            HasMemory, MemoryKind,
            Machine};

pub struct EvalContext<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    /// Stores the `Machine` instance.
    pub machine: M,

    /// The results of the type checker, from rustc.
    pub tcx: TyCtxtAt<'a, 'tcx, 'tcx>,

    /// Bounds in scope for polymorphic evaluations.
    pub param_env: ty::ParamEnv<'tcx>,

    /// The virtual memory system.
    pub memory: Memory<'a, 'mir, 'tcx, M>,

    /// The virtual call stack.
    pub(crate) stack: Vec<Frame<'mir, 'tcx>>,

    /// The maximum number of stack frames allowed
    pub(crate) stack_limit: usize,

    /// The maximum number of terminators that may be evaluated.
    /// This prevents infinite loops and huge computations from freezing up const eval.
    /// Remove once halting problem is solved.
    pub(crate) terminators_remaining: usize,
}

/// A stack frame.
pub struct Frame<'mir, 'tcx: 'mir> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub mir: &'mir mir::Mir<'tcx>,

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
    /// `[return_ptr, arguments..., variables..., temporaries...]`. The locals are stored as `Option<Value>`s.
    /// `None` represents a local that is currently dead, while a live local
    /// can either directly contain `Scalar` or refer to some part of an `Allocation`.
    ///
    /// Before being initialized, arguments are `Value::Scalar(Scalar::undef())` and other locals are `None`.
    pub locals: IndexVec<mir::Local, Option<Value>>,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////
    /// The block that is currently executed (or will be executed after the above call stacks
    /// return).
    pub block: mir::BasicBlock,

    /// The index of the currently evaluated statement.
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

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for &'a EvalContext<'a, 'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'c, 'b, 'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout
    for &'c &'b mut EvalContext<'a, 'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> layout::HasTyCtxt<'tcx> for &'a EvalContext<'a, 'mir, 'tcx, M> {
    #[inline]
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'tcx, 'tcx> {
        *self.tcx
    }
}

impl<'c, 'b, 'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> layout::HasTyCtxt<'tcx>
    for &'c &'b mut EvalContext<'a, 'mir, 'tcx, M> {
    #[inline]
    fn tcx<'d>(&'d self) -> TyCtxt<'d, 'tcx, 'tcx> {
        *self.tcx
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> LayoutOf for &'a EvalContext<'a, 'mir, 'tcx, M> {
    type Ty = Ty<'tcx>;
    type TyLayout = EvalResult<'tcx, TyLayout<'tcx>>;

    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.tcx.layout_of(self.param_env.and(ty))
            .map_err(|layout| EvalErrorKind::Layout(layout).into())
    }
}

impl<'c, 'b, 'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> LayoutOf
    for &'c &'b mut EvalContext<'a, 'mir, 'tcx, M> {
    type Ty = Ty<'tcx>;
    type TyLayout = EvalResult<'tcx, TyLayout<'tcx>>;

    #[inline]
    fn layout_of(self, ty: Ty<'tcx>) -> Self::TyLayout {
        (&**self).layout_of(ty)
    }
}

const MAX_TERMINATORS: usize = 1_000_000;

impl<'a, 'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn new(
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        machine: M,
        memory_data: M::MemoryData,
    ) -> Self {
        EvalContext {
            machine,
            tcx,
            param_env,
            memory: Memory::new(tcx, memory_data),
            stack: Vec::new(),
            stack_limit: tcx.sess.const_eval_stack_frame_limit,
            terminators_remaining: MAX_TERMINATORS,
        }
    }

    pub(crate) fn with_fresh_body<F: FnOnce(&mut Self) -> R, R>(&mut self, f: F) -> R {
        let stack = mem::replace(&mut self.stack, Vec::new());
        let terminators_remaining = mem::replace(&mut self.terminators_remaining, MAX_TERMINATORS);
        let r = f(self);
        self.stack = stack;
        self.terminators_remaining = terminators_remaining;
        r
    }

    pub fn alloc_ptr(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, Pointer> {
        let layout = self.layout_of(ty)?;
        assert!(!layout.is_unsized(), "cannot alloc memory for unsized type");

        self.memory.allocate(layout.size, layout.align, Some(MemoryKind::Stack))
    }

    pub fn memory(&self) -> &Memory<'a, 'mir, 'tcx, M> {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut Memory<'a, 'mir, 'tcx, M> {
        &mut self.memory
    }

    pub fn stack(&self) -> &[Frame<'mir, 'tcx>] {
        &self.stack
    }

    #[inline]
    pub fn cur_frame(&self) -> usize {
        assert!(self.stack.len() > 0);
        self.stack.len() - 1
    }

    pub fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        let ptr = self.memory.allocate_bytes(s.as_bytes());
        Ok(Scalar::Ptr(ptr).to_value_with_len(s.len() as u64, self.tcx.tcx))
    }

    pub fn const_value_to_value(
        &mut self,
        val: ConstValue<'tcx>,
        _ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        match val {
            ConstValue::ByRef(alloc, offset) => {
                // FIXME: Allocate new AllocId for all constants inside
                let id = self.memory.allocate_value(alloc.clone(), Some(MemoryKind::Stack))?;
                Ok(Value::ByRef(Pointer::new(id, offset).into(), alloc.align))
            },
            ConstValue::ScalarPair(a, b) => Ok(Value::ScalarPair(a, b)),
            ConstValue::Scalar(val) => Ok(Value::Scalar(val)),
        }
    }

    pub(super) fn const_to_value(
        &mut self,
        const_val: &ConstVal<'tcx>,
        ty: Ty<'tcx>
    ) -> EvalResult<'tcx, Value> {
        match *const_val {
            ConstVal::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                self.read_global_as_value(GlobalId {
                    instance,
                    promoted: None,
                }, ty)
            }
            ConstVal::Value(val) => self.const_value_to_value(val, ty)
        }
    }

    pub(super) fn resolve(&self, def_id: DefId, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        trace!("resolve: {:?}, {:#?}", def_id, substs);
        trace!("substs: {:#?}", self.substs());
        trace!("param_env: {:#?}", self.param_env);
        let substs = self.tcx.subst_and_normalize_erasing_regions(
            self.substs(),
            self.param_env,
            &substs,
        );
        ty::Instance::resolve(
            *self.tcx,
            self.param_env,
            def_id,
            substs,
        ).ok_or_else(|| EvalErrorKind::TypeckError.into()) // turn error prop into a panic to expose associated type in const issue
    }

    pub(super) fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx, self.param_env)
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
                self.tcx.maybe_optimized_mir(def_id).ok_or_else(||
                    EvalErrorKind::NoMirFor(self.tcx.item_path_str(def_id)).into()
                )
            }
            _ => Ok(self.tcx.instance_mir(instance)),
        }
    }

    pub fn monomorphize(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // miri doesn't care about lifetimes, and will choke on some crazy ones
        // let's simply get rid of them
        let substituted = ty.subst(*self.tcx, substs);
        self.tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), substituted)
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
        mir: &'mir mir::Mir<'tcx>,
        return_place: Place,
        return_to_block: StackPopCleanup,
    ) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation += 1;

        let locals = if mir.local_decls.len() > 1 {
            let mut locals = IndexVec::from_elem(Some(Value::Scalar(Scalar::undef())), &mir.local_decls);
            match self.tcx.describe_def(instance.def_id()) {
                // statics and constants don't have `Storage*` statements, no need to look for them
                Some(Def::Static(..)) | Some(Def::Const(..)) | Some(Def::AssociatedConst(..)) => {},
                _ => {
                    trace!("push_stack_frame: {:?}: num_bbs: {}", span, mir.basic_blocks().len());
                    for block in mir.basic_blocks() {
                        for stmt in block.statements.iter() {
                            use rustc::mir::StatementKind::{StorageDead, StorageLive};
                            match stmt.kind {
                                StorageLive(local) |
                                StorageDead(local) => locals[local] = None,
                                _ => {}
                            }
                        }
                    }
                },
            }
            locals
        } else {
            // don't allocate at all for trivial constants
            IndexVec::new()
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
                    self.memory.mark_static_initialized(
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
                self.intrinsic_overflowing(
                    bin_op,
                    left,
                    right,
                    dest,
                    dest_ty,
                )?;
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
                let val = self.eval_operand_to_scalar(operand)?;
                let val = self.unary_op(un_op, val, dest_ty)?;
                self.write_scalar(
                    dest,
                    val,
                    dest_ty,
                )?;
            }

            Aggregate(ref kind, ref operands) => {
                self.inc_step_counter_and_check_limit(operands.len());

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
                    ty::TyArray(elem_ty, n) => (elem_ty, n.unwrap_usize(self.tcx.tcx)),
                    _ => {
                        bug!(
                            "tried to assign array-repeat to non-array type {:?}",
                            dest_ty
                        )
                    }
                };
                let elem_size = self.layout_of(elem_ty)?.size;
                let value = self.eval_operand(operand)?.value;

                let (dest, dest_align) = self.force_allocation(dest)?.to_ptr_align();

                // FIXME: speed up repeat filling
                for i in 0..length {
                    let elem_dest = dest.ptr_offset(elem_size * i as u64, &self)?;
                    self.write_value_to_ptr(value, elem_dest, dest_align, elem_ty)?;
                }
            }

            Len(ref place) => {
                // FIXME(CTFE): don't allow computing the length of arrays in const eval
                let src = self.eval_place(place)?;
                let ty = self.place_ty(place);
                let (_, len) = src.elem_ty_and_len(ty, self.tcx.tcx);
                let defined = self.memory.pointer_size().bits() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: len as u128,
                        defined,
                    },
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
                    PlaceExtra::Length(len) => ptr.to_value_with_len(len, self.tcx.tcx),
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
                let defined = self.memory.pointer_size().bits() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: layout.size.bytes() as u128,
                        defined,
                    },
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
                                // pointers to extern types
                                (Value::Scalar(_),_) |
                                // slices and trait objects to other slices/trait objects
                                (Value::ScalarPair(..), true) => {
                                    let valty = ValTy {
                                        value: src.value,
                                        ty: dest_ty,
                                    };
                                    self.write_value(valty, dest)?;
                                }
                                // slices and trait objects to thin pointers (dropping the metadata)
                                (Value::ScalarPair(data, _), false) => {
                                    let valty = ValTy {
                                        value: Value::Scalar(data),
                                        ty: dest_ty,
                                    };
                                    self.write_value(valty, dest)?;
                                }
                            }
                        } else {
                            let src_layout = self.layout_of(src.ty)?;
                            match src_layout.variants {
                                layout::Variants::Single { index } => {
                                    if let Some(def) = src.ty.ty_adt_def() {
                                        let discr_val = def
                                            .discriminant_for_variant(*self.tcx, index)
                                            .val;
                                        let defined = self
                                            .layout_of(dest_ty)
                                            .unwrap()
                                            .size
                                            .bits() as u8;
                                        return self.write_scalar(
                                            dest,
                                            Scalar::Bits {
                                                bits: discr_val,
                                                defined,
                                            },
                                            dest_ty);
                                    }
                                }
                                layout::Variants::Tagged { .. } |
                                layout::Variants::NicheFilling { .. } => {},
                            }

                            let src_val = self.value_to_scalar(src)?;
                            let dest_val = self.cast_scalar(src_val, src.ty, dest_ty)?;
                            let valty = ValTy {
                                value: Value::Scalar(dest_val),
                                ty: dest_ty,
                            };
                            self.write_value(valty, dest)?;
                        }
                    }

                    ReifyFnPointer => {
                        match self.eval_operand(operand)?.ty.sty {
                            ty::TyFnDef(def_id, substs) => {
                                if self.tcx.has_attr(def_id, "rustc_args_required_const") {
                                    bug!("reifying a fn ptr that requires \
                                          const arguments");
                                }
                                let instance: EvalResult<'tcx, _> = ty::Instance::resolve(
                                    *self.tcx,
                                    self.param_env,
                                    def_id,
                                    substs,
                                ).ok_or_else(|| EvalErrorKind::TypeckError.into());
                                let fn_ptr = self.memory.create_fn_alloc(instance?);
                                let valty = ValTy {
                                    value: Value::Scalar(fn_ptr.into()),
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
                                let substs = self.tcx.subst_and_normalize_erasing_regions(
                                    self.substs(),
                                    ty::ParamEnv::reveal_all(),
                                    &substs,
                                );
                                let instance = ty::Instance::resolve_closure(
                                    *self.tcx,
                                    def_id,
                                    substs,
                                    ty::ClosureKind::FnOnce,
                                );
                                let fn_ptr = self.memory.create_fn_alloc(instance);
                                let valty = ValTy {
                                    value: Value::Scalar(fn_ptr.into()),
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
                let defined = self.layout_of(dest_ty).unwrap().size.bits() as u8;
                self.write_scalar(dest, Scalar::Bits {
                    bits: discr_val,
                    defined,
                }, dest_ty)?;
            }
        }

        self.dump_local(dest);

        Ok(())
    }

    pub(super) fn type_is_fat_ptr(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
            ty::TyRef(_, ty, _) => !self.type_is_sized(ty),
            ty::TyAdt(def, _) if def.is_box() => !self.type_is_sized(ty.boxed_ty()),
            _ => false,
        }
    }

    pub(super) fn eval_operand_to_scalar(
        &mut self,
        op: &mir::Operand<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        let valty = self.eval_operand(op)?;
        self.value_to_scalar(valty)
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
        let ty = self.monomorphize(op.ty(self.mir(), *self.tcx), self.substs());
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
                        let instance = self.frame().instance;
                        self.read_global_as_value(GlobalId {
                            instance,
                            promoted: Some(index),
                        }, ty)?
                    }
                };

                Ok(ValTy {
                    value,
                    ty,
                })
            }
        }
    }

    /// reads a tag and produces the corresponding variant index
    pub fn read_discriminant_as_variant_index(
        &mut self,
        place: Place,
        ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, usize> {
        let layout = self.layout_of(ty)?;
        match layout.variants {
            ty::layout::Variants::Single { index } => Ok(index),
            ty::layout::Variants::Tagged { .. } => {
                let discr_val = self.read_discriminant_value(place, ty)?;
                ty
                    .ty_adt_def()
                    .expect("tagged layout for non adt")
                    .discriminants(self.tcx.tcx)
                    .position(|var| var.val == discr_val)
                    .ok_or_else(|| EvalErrorKind::InvalidDiscriminant.into())
            }
            ty::layout::Variants::NicheFilling { .. } => {
                let discr_val = self.read_discriminant_value(place, ty)?;
                assert_eq!(discr_val as usize as u128, discr_val);
                Ok(discr_val as usize)
            },
        }
    }

    pub fn read_discriminant_value(
        &mut self,
        place: Place,
        ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, u128> {
        let layout = self.layout_of(ty)?;
        trace!("read_discriminant_value {:#?}", layout);
        if layout.abi == layout::Abi::Uninhabited {
            return Ok(0);
        }

        match layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = ty.ty_adt_def().map_or(
                    index as u128,
                    |def| def.discriminant_for_variant(*self.tcx, index).val);
                return Ok(discr_val);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }

        let (discr_place, discr) = self.place_field(place, mir::Field::new(0), layout)?;
        trace!("discr place: {:?}, {:?}", discr_place, discr);
        let raw_discr = self.value_to_scalar(ValTy {
            value: self.read_place(discr_place)?,
            ty: discr.ty
        })?;
        let discr_val = match layout.variants {
            layout::Variants::Single { .. } => bug!(),
            // FIXME: should we catch invalid discriminants here?
            layout::Variants::Tagged { .. } => {
                if discr.ty.is_signed() {
                    let i = raw_discr.to_bits(discr.size)? as i128;
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the layout discriminant
                    let shift = 128 - discr.size.bits();
                    let sexted = (i << shift) >> shift;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = ty
                        .ty_adt_def().expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let discr_ty = layout::Integer::from_attr(self.tcx.tcx, discr_ty);
                    let shift = 128 - discr_ty.size().bits();
                    let truncatee = sexted as u128;
                    (truncatee << shift) >> shift
                } else {
                    raw_discr.to_bits(discr.size)?
                }
            },
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let variants_start = *niche_variants.start() as u128;
                let variants_end = *niche_variants.end() as u128;
                match raw_discr {
                    Scalar::Ptr(_) => {
                        assert!(niche_start == 0);
                        assert!(variants_start == variants_end);
                        dataful_variant as u128
                    },
                    Scalar::Bits { bits: raw_discr, defined } => {
                        if defined < discr.size.bits() as u8 {
                            return err!(ReadUndefBytes);
                        }
                        let discr = raw_discr.wrapping_sub(niche_start)
                            .wrapping_add(variants_start);
                        if variants_start <= discr && discr <= variants_end {
                            discr
                        } else {
                            dataful_variant as u128
                        }
                    },
                }
            }
        };

        Ok(discr_val)
    }


    pub fn write_discriminant_value(
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
            layout::Variants::Tagged { ref tag, .. } => {
                let discr_val = dest_ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(*self.tcx, variant_index)
                    .val;

                // raw discriminants for enums are isize or bigger during
                // their computation, but the in-memory tag is the smallest possible
                // representation
                let size = tag.value.size(self.tcx.tcx).bits();
                let shift = 128 - size;
                let discr_val = (discr_val << shift) >> shift;

                let (discr_dest, tag) = self.place_field(dest, mir::Field::new(0), layout)?;
                self.write_scalar(discr_dest, Scalar::Bits {
                    bits: discr_val,
                    defined: size as u8,
                }, tag.ty)?;
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
                    let niche_value = ((variant_index - niche_variants.start()) as u128)
                        .wrapping_add(niche_start);
                    self.write_scalar(niche_dest, Scalar::Bits {
                        bits: niche_value,
                        defined: niche.size.bits() as u8,
                    }, niche.ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn read_global_as_value(&mut self, gid: GlobalId<'tcx>, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if self.tcx.is_static(gid.instance.def_id()).is_some() {
            let alloc_id = self
                .tcx
                .alloc_map
                .lock()
                .intern_static(gid.instance.def_id());
            let layout = self.layout_of(ty)?;
            return Ok(Value::ByRef(Scalar::Ptr(alloc_id.into()), layout.align))
        }
        let cv = self.const_eval(gid)?;
        self.const_to_value(&cv.val, ty)
    }

    pub fn const_eval(&self, gid: GlobalId<'tcx>) -> EvalResult<'tcx, &'tcx ty::Const<'tcx>> {
        let param_env = if self.tcx.is_static(gid.instance.def_id()).is_some() {
            ty::ParamEnv::reveal_all()
        } else {
            self.param_env
        };
        self.tcx.const_eval(param_env.and(gid)).map_err(|err| EvalErrorKind::ReferencedConstant(err).into())
    }

    pub fn force_allocation(&mut self, place: Place) -> EvalResult<'tcx, Place> {
        let new_place = match place {
            Place::Local { frame, local } => {
                match self.stack[frame].locals[local] {
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
                        self.stack[frame].locals[local] =
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

    pub fn value_to_scalar(
        &self,
        ValTy { value, ty } : ValTy<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        match self.follow_by_ref_value(value, ty)? {
            Value::ByRef { .. } => bug!("follow_by_ref_value can't result in `ByRef`"),

            Value::Scalar(scalar) => {
                // TODO: Do we really want insta-UB here?
                self.ensure_valid_value(scalar, ty)?;
                Ok(scalar)
            }

            Value::ScalarPair(..) => bug!("value_to_scalar can't work with fat pointers"),
        }
    }

    pub fn write_ptr(&mut self, dest: Place, val: Scalar, dest_ty: Ty<'tcx>) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: val.to_value(),
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    pub fn write_scalar(
        &mut self,
        dest: Place,
        val: Scalar,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: Value::Scalar(val),
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
        // In case `src_val` is a `ScalarPair`, we don't do any magic here to handle padding properly, which is only
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
                self.memory.copy(src_ptr, align.min(layout.align), dest_ptr, layout.align, layout.size, false)?;
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
        dest: Scalar,
        dest_align: Align,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let layout = self.layout_of(dest_ty)?;
        trace!("write_value_to_ptr: {:#?}, {}, {:#?}", value, dest_ty, layout);
        match value {
            Value::ByRef(ptr, align) => {
                self.memory.copy(ptr, align.min(layout.align), dest, dest_align.min(layout.align), layout.size, false)
            }
            Value::Scalar(scalar) => {
                let signed = match layout.abi {
                    layout::Abi::Scalar(ref scal) => match scal.value {
                        layout::Primitive::Int(_, signed) => signed,
                        _ => false,
                    },
                    _ => match scalar {
                        Scalar::Bits { defined: 0, .. } => false,
                        _ => bug!("write_value_to_ptr: invalid ByVal layout: {:#?}", layout),
                    }
                };
                self.memory.write_scalar(dest, dest_align, scalar, layout.size, signed)
            }
            Value::ScalarPair(a_val, b_val) => {
                trace!("write_value_to_ptr valpair: {:#?}", layout);
                let (a, b) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (&a.value, &b.value),
                    _ => bug!("write_value_to_ptr: invalid ScalarPair layout: {:#?}", layout)
                };
                let (a_size, b_size) = (a.size(&self), b.size(&self));
                let a_ptr = dest;
                let b_offset = a_size.abi_align(b.align(&self));
                let b_ptr = dest.ptr_offset(b_offset, &self)?.into();
                // TODO: What about signedess?
                self.memory.write_scalar(a_ptr, dest_align, a_val, a_size, false)?;
                self.memory.write_scalar(b_ptr, dest_align, b_val, b_size, false)
            }
        }
    }

    fn ensure_valid_value(&self, val: Scalar, ty: Ty<'tcx>) -> EvalResult<'tcx> {
        match ty.sty {
            ty::TyBool => val.to_bool().map(|_| ()),

            ty::TyChar if ::std::char::from_u32(val.to_bits(Size::from_bytes(4))? as u32).is_none() => {
                err!(InvalidChar(val.to_bits(Size::from_bytes(4))? as u32 as u128))
            }

            _ => Ok(()),
        }
    }

    pub fn read_value(&self, ptr: Scalar, align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, align, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    pub(crate) fn read_ptr(
        &self,
        ptr: Pointer,
        ptr_align: Align,
        pointee_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        let ptr_size = self.memory.pointer_size();
        let p: Scalar = self.memory.read_ptr_sized(ptr, ptr_align)?.into();
        if self.type_is_sized(pointee_ty) {
            Ok(p.to_value())
        } else {
            trace!("reading fat pointer extra of type {}", pointee_ty);
            let extra = ptr.offset(ptr_size, self)?;
            match self.tcx.struct_tail(pointee_ty).sty {
                ty::TyDynamic(..) => Ok(p.to_value_with_vtable(
                    self.memory.read_ptr_sized(extra, ptr_align)?.to_ptr()?,
                )),
                ty::TySlice(..) | ty::TyStr => {
                    let len = self
                        .memory
                        .read_ptr_sized(extra, ptr_align)?
                        .to_bits(ptr_size)?;
                    Ok(p.to_value_with_len(len as u64, self.tcx.tcx))
                },
                _ => bug!("unsized scalar ptr read from {:?}", pointee_ty),
            }
        }
    }

    pub fn validate_ptr_target(
        &self,
        ptr: Pointer,
        ptr_align: Align,
        ty: Ty<'tcx>
    ) -> EvalResult<'tcx> {
        match ty.sty {
            ty::TyBool => {
                self.memory.read_scalar(ptr, ptr_align, Size::from_bytes(1))?.to_bool()?;
            }
            ty::TyChar => {
                let c = self.memory.read_scalar(ptr, ptr_align, Size::from_bytes(4))?.to_bits(Size::from_bytes(4))? as u32;
                match ::std::char::from_u32(c) {
                    Some(..) => (),
                    None => return err!(InvalidChar(c as u128)),
                }
            }

            ty::TyFnPtr(_) => {
                self.memory.read_ptr_sized(ptr, ptr_align)?;
            },
            ty::TyRef(_, rty, _) |
            ty::TyRawPtr(ty::TypeAndMut { ty: rty, .. }) => {
                self.read_ptr(ptr, ptr_align, rty)?;
            }

            ty::TyAdt(def, _) => {
                if def.is_box() {
                    self.read_ptr(ptr, ptr_align, ty.boxed_ty())?;
                    return Ok(());
                }

                if let layout::Abi::Scalar(ref scalar) = self.layout_of(ty)?.abi {
                    let size = scalar.value.size(self);
                    self.memory.read_scalar(ptr, ptr_align, size)?;
                }
            }

            _ => (),
        }
        Ok(())
    }

    pub fn try_read_by_ref(&self, mut val: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        // Convert to ByVal or ScalarPair if possible
        if let Value::ByRef(ptr, align) = val {
            if let Some(read_val) = self.try_read_value(ptr, align, ty)? {
                val = read_val;
            }
        }
        Ok(val)
    }

    pub fn try_read_value(&self, ptr: Scalar, ptr_align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        let layout = self.layout_of(ty)?;
        self.memory.check_align(ptr, ptr_align)?;

        if layout.size.bytes() == 0 {
            return Ok(Some(Value::Scalar(Scalar::undef())));
        }

        let ptr = ptr.to_ptr()?;

        // Not the right place to do this
        //self.validate_ptr_target(ptr, ptr_align, ty)?;

        match layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory.read_scalar(ptr, ptr_align, layout.size)?;
                Ok(Some(Value::Scalar(scalar)))
            }
            layout::Abi::ScalarPair(ref a, ref b) => {
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.abi_align(b.align(self));
                let b_ptr = ptr.offset(b_offset, self)?.into();
                let a_val = self.memory.read_scalar(a_ptr, ptr_align, a_size)?;
                let b_val = self.memory.read_scalar(b_ptr, ptr_align, b_size)?;
                Ok(Some(Value::ScalarPair(a_val, b_val)))
            }
            _ => Ok(None),
        }
    }

    pub fn frame(&self) -> &Frame<'mir, 'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    pub fn frame_mut(&mut self) -> &mut Frame<'mir, 'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    pub(super) fn mir(&self) -> &'mir mir::Mir<'tcx> {
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
                    value: ptr.to_value_with_len(length.unwrap_usize(self.tcx.tcx), self.tcx.tcx),
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
                    *self.tcx,
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
            (&ty::TyRef(_, s, _), &ty::TyRef(_, d, _)) |
            (&ty::TyRef(_, s, _), &ty::TyRawPtr(TypeAndMut { ty: d, .. })) |
            (&ty::TyRawPtr(TypeAndMut { ty: s, .. }),
             &ty::TyRawPtr(TypeAndMut { ty: d, .. })) => {
                self.unsize_into_ptr(src, src_layout.ty, dst, dst_layout.ty, s, d)
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
                            let src_place = Place::from_scalar_ptr(ptr, align);
                            let (src_f_place, src_field) =
                                self.place_field(src_place, mir::Field::new(i), src_layout)?;
                            (self.read_place(src_f_place)?, src_field)
                        }
                        Value::Scalar(_) | Value::ScalarPair(..) => {
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
        if !log_enabled!(::log::Level::Trace) {
            return;
        }
        match place {
            Place::Local { frame, local } => {
                let mut allocs = Vec::new();
                let mut msg = format!("{:?}", local);
                if frame != self.cur_frame() {
                    write!(msg, " ({} frames up)", self.cur_frame() - frame).unwrap();
                }
                write!(msg, ":").unwrap();

                match self.stack[frame].get_local(local) {
                    Err(err) => {
                        if let EvalErrorKind::DeadLocal = err.kind {
                            write!(msg, " is dead").unwrap();
                        } else {
                            panic!("Failed to access local: {:?}", err);
                        }
                    }
                    Ok(Value::ByRef(ptr, align)) => {
                        match ptr {
                            Scalar::Ptr(ptr) => {
                                write!(msg, " by align({}) ref:", align.abi()).unwrap();
                                allocs.push(ptr.alloc_id);
                            }
                            ptr => write!(msg, " integral by ref: {:?}", ptr).unwrap(),
                        }
                    }
                    Ok(Value::Scalar(val)) => {
                        write!(msg, " {:?}", val).unwrap();
                        if let Scalar::Ptr(ptr) = val {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                    Ok(Value::ScalarPair(val1, val2)) => {
                        write!(msg, " ({:?}, {:?})", val1, val2).unwrap();
                        if let Scalar::Ptr(ptr) = val1 {
                            allocs.push(ptr.alloc_id);
                        }
                        if let Scalar::Ptr(ptr) = val2 {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                }

                trace!("{}", msg);
                self.memory.dump_allocs(allocs);
            }
            Place::Ptr { ptr, align, .. } => {
                match ptr {
                    Scalar::Ptr(ptr) => {
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

    pub fn generate_stacktrace(&self, explicit_span: Option<Span>) -> (Vec<FrameInfo>, Span) {
        let mut last_span = None;
        let mut frames = Vec::new();
        // skip 1 because the last frame is just the environment of the constant
        for &Frame { instance, span, mir, block, stmt, .. } in self.stack().iter().skip(1).rev() {
            // make sure we don't emit frames that are duplicates of the previous
            if explicit_span == Some(span) {
                last_span = Some(span);
                continue;
            }
            if let Some(last) = last_span {
                if last == span {
                    continue;
                }
            } else {
                last_span = Some(span);
            }
            let location = if self.tcx.def_key(instance.def_id()).disambiguated_data.data == DefPathData::ClosureExpr {
                "closure".to_owned()
            } else {
                instance.to_string()
            };
            let block = &mir.basic_blocks()[block];
            let source_info = if stmt < block.statements.len() {
                block.statements[stmt].source_info
            } else {
                block.terminator().source_info
            };
            let lint_root = match mir.source_scope_local_data {
                mir::ClearCrossCrate::Set(ref ivs) => Some(ivs[source_info.scope].lint_root),
                mir::ClearCrossCrate::Clear => None,
            };
            frames.push(FrameInfo { span, location, lint_root });
        }
        trace!("generate stacktrace: {:#?}, {:?}", frames, explicit_span);
        (frames, self.tcx.span)
    }

    pub fn sign_extend(&self, value: u128, ty: Ty<'tcx>) -> EvalResult<'tcx, u128> {
        super::sign_extend(self.tcx.tcx, value, ty)
    }

    pub fn truncate(&self, value: u128, ty: Ty<'tcx>) -> EvalResult<'tcx, u128> {
        super::truncate(self.tcx.tcx, value, ty)
    }
}

impl<'mir, 'tcx> Frame<'mir, 'tcx> {
    pub fn get_local(&self, local: mir::Local) -> EvalResult<'tcx, Value> {
        self.locals[local].ok_or_else(|| EvalErrorKind::DeadLocal.into())
    }

    fn set_local(&mut self, local: mir::Local, value: Value) -> EvalResult<'tcx> {
        match self.locals[local] {
            None => err!(DeadLocal),
            Some(ref mut local) => {
                *local = value;
                Ok(())
            }
        }
    }

    pub fn storage_live(&mut self, local: mir::Local) -> Option<Value> {
        trace!("{:?} is now live", local);

        // StorageLive *always* kills the value that's currently stored
        mem::replace(&mut self.locals[local], Some(Value::Scalar(Scalar::undef())))
    }

    /// Returns the old value of the local
    pub fn storage_dead(&mut self, local: mir::Local) -> Option<Value> {
        trace!("{:?} is now dead", local);

        self.locals[local].take()
    }
}
