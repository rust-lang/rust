use std::cell::Cell;
use std::fmt::Write;
use std::mem;

use rustc::ich::StableHashingContext;
use rustc::mir;
use rustc::mir::interpret::{
    sign_extend, truncate, AllocId, FrameInfo, GlobalId, InterpResult, Pointer, Scalar,
};
use rustc::ty::layout::{self, Align, HasDataLayout, LayoutOf, Size, TyLayout};
use rustc::ty::query::TyCtxtAt;
use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::vec::IndexVec;
use rustc_macros::HashStable;
use rustc_span::source_map::{self, Span, DUMMY_SP};

use super::{
    Immediate, MPlaceTy, Machine, MemPlace, MemPlaceMeta, Memory, OpTy, Operand, Place, PlaceTy,
    ScalarMaybeUndef, StackPopInfo,
};

pub struct InterpCx<'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// Stores the `Machine` instance.
    pub machine: M,

    /// The results of the type checker, from rustc.
    pub tcx: TyCtxtAt<'tcx>,

    /// Bounds in scope for polymorphic evaluations.
    pub(crate) param_env: ty::ParamEnv<'tcx>,

    /// The virtual memory system.
    pub memory: Memory<'mir, 'tcx, M>,

    /// The virtual call stack.
    pub(crate) stack: Vec<Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra>>,

    /// A cache for deduplicating vtables
    pub(super) vtables:
        FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), Pointer<M::PointerTag>>,
}

/// A stack frame.
#[derive(Clone)]
pub struct Frame<'mir, 'tcx, Tag = (), Extra = ()> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub body: &'mir mir::Body<'tcx>,

    /// The def_id and substs of the current function.
    pub instance: ty::Instance<'tcx>,

    /// The span of the call site.
    pub span: source_map::Span,

    /// Extra data for the machine.
    pub extra: Extra,

    ////////////////////////////////////////////////////////////////////////////////
    // Return place and locals
    ////////////////////////////////////////////////////////////////////////////////
    /// Work to perform when returning from this function.
    pub return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to,
    /// and its layout in the caller.
    pub return_place: Option<PlaceTy<'tcx, Tag>>,

    /// The list of locals for this stack frame, stored in order as
    /// `[return_ptr, arguments..., variables..., temporaries...]`.
    /// The locals are stored as `Option<Value>`s.
    /// `None` represents a local that is currently dead, while a live local
    /// can either directly contain `Scalar` or refer to some part of an `Allocation`.
    pub locals: IndexVec<mir::Local, LocalState<'tcx, Tag>>,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////
    /// The block that is currently executed (or will be executed after the above call stacks
    /// return).
    /// If this is `None`, we are unwinding and this function doesn't need any clean-up.
    /// Just continue the same as with `Resume`.
    pub block: Option<mir::BasicBlock>,

    /// The index of the currently evaluated statement.
    pub stmt: usize,
}

#[derive(Clone, Eq, PartialEq, Debug, HashStable)] // Miri debug-prints these
pub enum StackPopCleanup {
    /// Jump to the next block in the caller, or cause UB if None (that's a function
    /// that may never return). Also store layout of return place so
    /// we can validate it at that layout.
    /// `ret` stores the block we jump to on a normal return, while 'unwind'
    /// stores the block used for cleanup during unwinding
    Goto { ret: Option<mir::BasicBlock>, unwind: Option<mir::BasicBlock> },
    /// Just do nohing: Used by Main and for the box_alloc hook in miri.
    /// `cleanup` says whether locals are deallocated. Static computation
    /// wants them leaked to intern what they need (and just throw away
    /// the entire `ecx` when it is done).
    None { cleanup: bool },
}

/// State of a local variable including a memoized layout
#[derive(Clone, PartialEq, Eq, HashStable)]
pub struct LocalState<'tcx, Tag = (), Id = AllocId> {
    pub value: LocalValue<Tag, Id>,
    /// Don't modify if `Some`, this is only used to prevent computing the layout twice
    #[stable_hasher(ignore)]
    pub layout: Cell<Option<TyLayout<'tcx>>>,
}

/// Current value of a local variable
#[derive(Copy, Clone, PartialEq, Eq, Debug, HashStable)] // Miri debug-prints these
pub enum LocalValue<Tag = (), Id = AllocId> {
    /// This local is not currently alive, and cannot be used at all.
    Dead,
    /// This local is alive but not yet initialized. It can be written to
    /// but not read from or its address taken. Locals get initialized on
    /// first write because for unsized locals, we do not know their size
    /// before that.
    Uninitialized,
    /// A normal, live local.
    /// Mostly for convenience, we re-use the `Operand` type here.
    /// This is an optimization over just always having a pointer here;
    /// we can thus avoid doing an allocation when the local just stores
    /// immediate values *and* never has its address taken.
    Live(Operand<Tag, Id>),
}

impl<'tcx, Tag: Copy + 'static> LocalState<'tcx, Tag> {
    pub fn access(&self) -> InterpResult<'tcx, Operand<Tag>> {
        match self.value {
            LocalValue::Dead => throw_unsup!(DeadLocal),
            LocalValue::Uninitialized => {
                bug!("The type checker should prevent reading from a never-written local")
            }
            LocalValue::Live(val) => Ok(val),
        }
    }

    /// Overwrite the local.  If the local can be overwritten in place, return a reference
    /// to do so; otherwise return the `MemPlace` to consult instead.
    pub fn access_mut(
        &mut self,
    ) -> InterpResult<'tcx, Result<&mut LocalValue<Tag>, MemPlace<Tag>>> {
        match self.value {
            LocalValue::Dead => throw_unsup!(DeadLocal),
            LocalValue::Live(Operand::Indirect(mplace)) => Ok(Err(mplace)),
            ref mut local @ LocalValue::Live(Operand::Immediate(_))
            | ref mut local @ LocalValue::Uninitialized => Ok(Ok(local)),
        }
    }
}

impl<'mir, 'tcx, Tag, Extra> Frame<'mir, 'tcx, Tag, Extra> {
    /// Return the `SourceInfo` of the current instruction.
    pub fn current_source_info(&self) -> Option<mir::SourceInfo> {
        self.block.map(|block| {
            let block = &self.body.basic_blocks()[block];
            if self.stmt < block.statements.len() {
                block.statements[self.stmt].source_info
            } else {
                block.terminator().source_info
            }
        })
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for InterpCx<'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &layout::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'mir, 'tcx, M> layout::HasTyCtxt<'tcx> for InterpCx<'mir, 'tcx, M>
where
    M: Machine<'mir, 'tcx>,
{
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        *self.tcx
    }
}

impl<'mir, 'tcx, M> layout::HasParamEnv<'tcx> for InterpCx<'mir, 'tcx, M>
where
    M: Machine<'mir, 'tcx>,
{
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> LayoutOf for InterpCx<'mir, 'tcx, M> {
    type Ty = Ty<'tcx>;
    type TyLayout = InterpResult<'tcx, TyLayout<'tcx>>;

    #[inline]
    fn layout_of(&self, ty: Ty<'tcx>) -> Self::TyLayout {
        self.tcx
            .layout_of(self.param_env.and(ty))
            .map_err(|layout| err_inval!(Layout(layout)).into())
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    pub fn new(
        tcx: TyCtxtAt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        machine: M,
        memory_extra: M::MemoryExtra,
    ) -> Self {
        InterpCx {
            machine,
            tcx,
            param_env,
            memory: Memory::new(tcx, memory_extra),
            stack: Vec::new(),
            vtables: FxHashMap::default(),
        }
    }

    #[inline(always)]
    pub fn force_ptr(
        &self,
        scalar: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        self.memory.force_ptr(scalar)
    }

    #[inline(always)]
    pub fn force_bits(
        &self,
        scalar: Scalar<M::PointerTag>,
        size: Size,
    ) -> InterpResult<'tcx, u128> {
        self.memory.force_bits(scalar, size)
    }

    /// Call this to turn untagged "global" pointers (obtained via `tcx`) into
    /// the *canonical* machine pointer to the allocation.  Must never be used
    /// for any other pointers!
    ///
    /// This represents a *direct* access to that memory, as opposed to access
    /// through a pointer that was created by the program.
    #[inline(always)]
    pub fn tag_static_base_pointer(&self, ptr: Pointer) -> Pointer<M::PointerTag> {
        self.memory.tag_static_base_pointer(ptr)
    }

    #[inline(always)]
    pub fn stack(&self) -> &[Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra>] {
        &self.stack
    }

    #[inline(always)]
    pub fn cur_frame(&self) -> usize {
        assert!(self.stack.len() > 0);
        self.stack.len() - 1
    }

    #[inline(always)]
    pub fn frame(&self) -> &Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra> {
        self.stack.last().expect("no call frames exist")
    }

    #[inline(always)]
    pub fn frame_mut(&mut self) -> &mut Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra> {
        self.stack.last_mut().expect("no call frames exist")
    }

    #[inline(always)]
    pub(super) fn body(&self) -> &'mir mir::Body<'tcx> {
        self.frame().body
    }

    #[inline(always)]
    pub fn sign_extend(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        assert!(ty.abi.is_signed());
        sign_extend(value, ty.size)
    }

    #[inline(always)]
    pub fn truncate(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        truncate(value, ty.size)
    }

    #[inline]
    pub fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx, self.param_env)
    }

    #[inline]
    pub fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        ty.is_freeze(*self.tcx, self.param_env, DUMMY_SP)
    }

    pub fn load_mir(
        &self,
        instance: ty::InstanceDef<'tcx>,
        promoted: Option<mir::Promoted>,
    ) -> InterpResult<'tcx, mir::ReadOnlyBodyAndCache<'tcx, 'tcx>> {
        // do not continue if typeck errors occurred (can only occur in local crate)
        let did = instance.def_id();
        if did.is_local()
            && self.tcx.has_typeck_tables(did)
            && self.tcx.typeck_tables_of(did).tainted_by_errors
        {
            throw_inval!(TypeckError)
        }
        trace!("load mir(instance={:?}, promoted={:?})", instance, promoted);
        if let Some(promoted) = promoted {
            return Ok(self.tcx.promoted_mir(did)[promoted].unwrap_read_only());
        }
        match instance {
            ty::InstanceDef::Item(def_id) => {
                if self.tcx.is_mir_available(did) {
                    Ok(self.tcx.optimized_mir(did).unwrap_read_only())
                } else {
                    throw_unsup!(NoMirFor(self.tcx.def_path_str(def_id)))
                }
            }
            _ => Ok(self.tcx.instance_mir(instance)),
        }
    }

    /// Call this on things you got out of the MIR (so it is as generic as the current
    /// stack frame), to bring it into the proper environment for this interpreter.
    pub(super) fn subst_from_frame_and_normalize_erasing_regions<T: TypeFoldable<'tcx>>(
        &self,
        value: T,
    ) -> T {
        self.tcx.subst_and_normalize_erasing_regions(
            self.frame().instance.substs,
            self.param_env,
            &value,
        )
    }

    /// The `substs` are assumed to already be in our interpreter "universe" (param_env).
    pub(super) fn resolve(
        &self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> InterpResult<'tcx, ty::Instance<'tcx>> {
        trace!("resolve: {:?}, {:#?}", def_id, substs);
        trace!("param_env: {:#?}", self.param_env);
        trace!("substs: {:#?}", substs);
        ty::Instance::resolve(*self.tcx, self.param_env, def_id, substs)
            .ok_or_else(|| err_inval!(TooGeneric).into())
    }

    pub fn layout_of_local(
        &self,
        frame: &Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, TyLayout<'tcx>> {
        // `const_prop` runs into this with an invalid (empty) frame, so we
        // have to support that case (mostly by skipping all caching).
        match frame.locals.get(local).and_then(|state| state.layout.get()) {
            None => {
                let layout = crate::interpret::operand::from_known_layout(layout, || {
                    let local_ty = frame.body.local_decls[local].ty;
                    let local_ty = self.tcx.subst_and_normalize_erasing_regions(
                        frame.instance.substs,
                        self.param_env,
                        &local_ty,
                    );
                    self.layout_of(local_ty)
                })?;
                if let Some(state) = frame.locals.get(local) {
                    // Layouts of locals are requested a lot, so we cache them.
                    state.layout.set(Some(layout));
                }
                Ok(layout)
            }
            Some(layout) => Ok(layout),
        }
    }

    /// Returns the actual dynamic size and alignment of the place at the given type.
    /// Only the "meta" (metadata) part of the place matters.
    /// This can fail to provide an answer for extern types.
    pub(super) fn size_and_align_of(
        &self,
        metadata: MemPlaceMeta<M::PointerTag>,
        layout: TyLayout<'tcx>,
    ) -> InterpResult<'tcx, Option<(Size, Align)>> {
        if !layout.is_unsized() {
            return Ok(Some((layout.size, layout.align.abi)));
        }
        match layout.ty.kind {
            ty::Adt(..) | ty::Tuple(..) => {
                // First get the size of all statically known fields.
                // Don't use type_of::sizing_type_of because that expects t to be sized,
                // and it also rounds up to alignment, which we want to avoid,
                // as the unsized field's alignment could be smaller.
                assert!(!layout.ty.is_simd());
                trace!("DST layout: {:?}", layout);

                let sized_size = layout.fields.offset(layout.fields.count() - 1);
                let sized_align = layout.align.abi;
                trace!(
                    "DST {} statically sized prefix size: {:?} align: {:?}",
                    layout.ty,
                    sized_size,
                    sized_align
                );

                // Recurse to get the size of the dynamically sized field (must be
                // the last field).  Can't have foreign types here, how would we
                // adjust alignment and size for them?
                let field = layout.field(self, layout.fields.count() - 1)?;
                let (unsized_size, unsized_align) = match self.size_and_align_of(metadata, field)? {
                    Some(size_and_align) => size_and_align,
                    None => {
                        // A field with extern type.  If this field is at offset 0, we behave
                        // like the underlying extern type.
                        // FIXME: Once we have made decisions for how to handle size and alignment
                        // of `extern type`, this should be adapted.  It is just a temporary hack
                        // to get some code to work that probably ought to work.
                        if sized_size == Size::ZERO {
                            return Ok(None);
                        } else {
                            bug!("Fields cannot be extern types, unless they are at offset 0")
                        }
                    }
                };

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
                let size = size.align_to(align);

                // Check if this brought us over the size limit.
                if size.bytes() >= self.tcx.data_layout().obj_size_bound() {
                    throw_ub_format!(
                        "wide pointer metadata contains invalid information: \
                        total size is bigger than largest supported object"
                    );
                }
                Ok(Some((size, align)))
            }
            ty::Dynamic(..) => {
                let vtable = metadata.unwrap_meta();
                // Read size and align from vtable (already checks size).
                Ok(Some(self.read_size_and_align_from_vtable(vtable)?))
            }

            ty::Slice(_) | ty::Str => {
                let len = metadata.unwrap_meta().to_machine_usize(self)?;
                let elem = layout.field(self, 0)?;

                // Make sure the slice is not too big.
                let size = elem.size.checked_mul(len, &*self.tcx).ok_or_else(|| {
                    err_ub_format!(
                        "invalid slice: \
                        total size is bigger than largest supported object"
                    )
                })?;
                Ok(Some((size, elem.align.abi)))
            }

            ty::Foreign(_) => Ok(None),

            _ => bug!("size_and_align_of::<{:?}> not supported", layout.ty),
        }
    }
    #[inline]
    pub fn size_and_align_of_mplace(
        &self,
        mplace: MPlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, Option<(Size, Align)>> {
        self.size_and_align_of(mplace.meta, mplace.layout)
    }

    pub fn push_stack_frame(
        &mut self,
        instance: ty::Instance<'tcx>,
        span: Span,
        body: &'mir mir::Body<'tcx>,
        return_place: Option<PlaceTy<'tcx, M::PointerTag>>,
        return_to_block: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        if self.stack.len() > 0 {
            info!("PAUSING({}) {}", self.cur_frame(), self.frame().instance);
        }
        ::log_settings::settings().indentation += 1;

        // first push a stack frame so we have access to the local substs
        let extra = M::stack_push(self)?;
        self.stack.push(Frame {
            body,
            block: Some(mir::START_BLOCK),
            return_to_block,
            return_place,
            // empty local array, we fill it in below, after we are inside the stack frame and
            // all methods actually know about the frame
            locals: IndexVec::new(),
            span,
            instance,
            stmt: 0,
            extra,
        });

        // don't allocate at all for trivial constants
        if body.local_decls.len() > 1 {
            // Locals are initially uninitialized.
            let dummy = LocalState { value: LocalValue::Uninitialized, layout: Cell::new(None) };
            let mut locals = IndexVec::from_elem(dummy, &body.local_decls);
            // Return place is handled specially by the `eval_place` functions, and the
            // entry in `locals` should never be used. Make it dead, to be sure.
            locals[mir::RETURN_PLACE].value = LocalValue::Dead;
            // Now mark those locals as dead that we do not want to initialize
            match self.tcx.def_kind(instance.def_id()) {
                // statics and constants don't have `Storage*` statements, no need to look for them
                Some(DefKind::Static) | Some(DefKind::Const) | Some(DefKind::AssocConst) => {}
                _ => {
                    trace!("push_stack_frame: {:?}: num_bbs: {}", span, body.basic_blocks().len());
                    for block in body.basic_blocks() {
                        for stmt in block.statements.iter() {
                            use rustc::mir::StatementKind::{StorageDead, StorageLive};
                            match stmt.kind {
                                StorageLive(local) | StorageDead(local) => {
                                    locals[local].value = LocalValue::Dead;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // done
            self.frame_mut().locals = locals;
        }

        info!("ENTERING({}) {}", self.cur_frame(), self.frame().instance);

        if self.stack.len() > *self.tcx.sess.recursion_limit.get() {
            throw_exhaust!(StackFrameLimitReached)
        } else {
            Ok(())
        }
    }

    /// Jump to the given block.
    #[inline]
    pub fn go_to_block(&mut self, target: mir::BasicBlock) {
        let frame = self.frame_mut();
        frame.block = Some(target);
        frame.stmt = 0;
    }

    /// *Return* to the given `target` basic block.
    /// Do *not* use for unwinding! Use `unwind_to_block` instead.
    ///
    /// If `target` is `None`, that indicates the function cannot return, so we raise UB.
    pub fn return_to_block(&mut self, target: Option<mir::BasicBlock>) -> InterpResult<'tcx> {
        if let Some(target) = target {
            Ok(self.go_to_block(target))
        } else {
            throw_ub!(Unreachable)
        }
    }

    /// *Unwind* to the given `target` basic block.
    /// Do *not* use for returning! Use `return_to_block` instead.
    ///
    /// If `target` is `None`, that indicates the function does not need cleanup during
    /// unwinding, and we will just keep propagating that upwards.
    pub fn unwind_to_block(&mut self, target: Option<mir::BasicBlock>) {
        let frame = self.frame_mut();
        frame.block = target;
        frame.stmt = 0;
    }

    /// Pops the current frame from the stack, deallocating the
    /// memory for allocated locals.
    ///
    /// If `unwinding` is `false`, then we are performing a normal return
    /// from a function. In this case, we jump back into the frame of the caller,
    /// and continue execution as normal.
    ///
    /// If `unwinding` is `true`, then we are in the middle of a panic,
    /// and need to unwind this frame. In this case, we jump to the
    /// `cleanup` block for the function, which is responsible for running
    /// `Drop` impls for any locals that have been initialized at this point.
    /// The cleanup block ends with a special `Resume` terminator, which will
    /// cause us to continue unwinding.
    pub(super) fn pop_stack_frame(&mut self, unwinding: bool) -> InterpResult<'tcx> {
        info!(
            "LEAVING({}) {} (unwinding = {})",
            self.cur_frame(),
            self.frame().instance,
            unwinding
        );

        // Sanity check `unwinding`.
        assert_eq!(
            unwinding,
            match self.frame().block {
                None => true,
                Some(block) => self.body().basic_blocks()[block].is_cleanup,
            }
        );

        ::log_settings::settings().indentation -= 1;
        let frame = self.stack.pop().expect("tried to pop a stack frame, but there were none");
        let stack_pop_info = M::stack_pop(self, frame.extra, unwinding)?;
        if let (false, StackPopInfo::StopUnwinding) = (unwinding, stack_pop_info) {
            bug!("Attempted to stop unwinding while there is no unwinding!");
        }

        // Now where do we jump next?

        // Determine if we leave this function normally or via unwinding.
        let cur_unwinding =
            if let StackPopInfo::StopUnwinding = stack_pop_info { false } else { unwinding };

        // Usually we want to clean up (deallocate locals), but in a few rare cases we don't.
        // In that case, we return early. We also avoid validation in that case,
        // because this is CTFE and the final value will be thoroughly validated anyway.
        let (cleanup, next_block) = match frame.return_to_block {
            StackPopCleanup::Goto { ret, unwind } => {
                (true, Some(if cur_unwinding { unwind } else { ret }))
            }
            StackPopCleanup::None { cleanup, .. } => (cleanup, None),
        };

        if !cleanup {
            assert!(self.stack.is_empty(), "only the topmost frame should ever be leaked");
            assert!(next_block.is_none(), "tried to skip cleanup when we have a next block!");
            // Leak the locals, skip validation.
            return Ok(());
        }

        // Cleanup: deallocate all locals that are backed by an allocation.
        for local in frame.locals {
            self.deallocate_local(local.value)?;
        }

        trace!(
            "StackPopCleanup: {:?} StackPopInfo: {:?} cur_unwinding = {:?}",
            frame.return_to_block,
            stack_pop_info,
            cur_unwinding
        );
        if cur_unwinding {
            // Follow the unwind edge.
            let unwind = next_block.expect("Encounted StackPopCleanup::None when unwinding!");
            self.unwind_to_block(unwind);
        } else {
            // Follow the normal return edge.
            // Validate the return value. Do this after deallocating so that we catch dangling
            // references.
            if let Some(return_place) = frame.return_place {
                if M::enforce_validity(self) {
                    // Data got changed, better make sure it matches the type!
                    // It is still possible that the return place held invalid data while
                    // the function is running, but that's okay because nobody could have
                    // accessed that same data from the "outside" to observe any broken
                    // invariant -- that is, unless a function somehow has a ptr to
                    // its return place... but the way MIR is currently generated, the
                    // return place is always a local and then this cannot happen.
                    self.validate_operand(self.place_to_op(return_place)?, vec![], None)?;
                }
            } else {
                // Uh, that shouldn't happen... the function did not intend to return
                throw_ub!(Unreachable);
            }

            // Jump to new block -- *after* validation so that the spans make more sense.
            if let Some(ret) = next_block {
                self.return_to_block(ret)?;
            }
        }

        if self.stack.len() > 0 {
            info!(
                "CONTINUING({}) {} (unwinding = {})",
                self.cur_frame(),
                self.frame().instance,
                cur_unwinding
            );
        }

        Ok(())
    }

    /// Mark a storage as live, killing the previous content and returning it.
    /// Remember to deallocate that!
    pub fn storage_live(
        &mut self,
        local: mir::Local,
    ) -> InterpResult<'tcx, LocalValue<M::PointerTag>> {
        assert!(local != mir::RETURN_PLACE, "Cannot make return place live");
        trace!("{:?} is now live", local);

        let local_val = LocalValue::Uninitialized;
        // StorageLive *always* kills the value that's currently stored.
        // However, we do not error if the variable already is live;
        // see <https://github.com/rust-lang/rust/issues/42371>.
        Ok(mem::replace(&mut self.frame_mut().locals[local].value, local_val))
    }

    /// Returns the old value of the local.
    /// Remember to deallocate that!
    pub fn storage_dead(&mut self, local: mir::Local) -> LocalValue<M::PointerTag> {
        assert!(local != mir::RETURN_PLACE, "Cannot make return place dead");
        trace!("{:?} is now dead", local);

        mem::replace(&mut self.frame_mut().locals[local].value, LocalValue::Dead)
    }

    pub(super) fn deallocate_local(
        &mut self,
        local: LocalValue<M::PointerTag>,
    ) -> InterpResult<'tcx> {
        // FIXME: should we tell the user that there was a local which was never written to?
        if let LocalValue::Live(Operand::Indirect(MemPlace { ptr, .. })) = local {
            trace!("deallocating local");
            // All locals have a backing allocation, even if the allocation is empty
            // due to the local having ZST type.
            let ptr = ptr.assert_ptr();
            if log_enabled!(::log::Level::Trace) {
                self.memory.dump_alloc(ptr.alloc_id);
            }
            self.memory.deallocate_local(ptr)?;
        };
        Ok(())
    }

    pub(super) fn const_eval(
        &self,
        gid: GlobalId<'tcx>,
        ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        // For statics we pick `ParamEnv::reveal_all`, because statics don't have generics
        // and thus don't care about the parameter environment. While we could just use
        // `self.param_env`, that would mean we invoke the query to evaluate the static
        // with different parameter environments, thus causing the static to be evaluated
        // multiple times.
        let param_env = if self.tcx.is_static(gid.instance.def_id()) {
            ty::ParamEnv::reveal_all()
        } else {
            self.param_env
        };
        let val = self.tcx.const_eval_global_id(param_env, gid, Some(self.tcx.span))?;

        // Even though `ecx.const_eval` is called from `eval_const_to_op` we can never have a
        // recursion deeper than one level, because the `tcx.const_eval` above is guaranteed to not
        // return `ConstValue::Unevaluated`, which is the only way that `eval_const_to_op` will call
        // `ecx.const_eval`.
        let const_ = ty::Const { val: ty::ConstKind::Value(val), ty };
        self.eval_const_to_op(&const_, None)
    }

    pub fn const_eval_raw(
        &self,
        gid: GlobalId<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::PointerTag>> {
        // For statics we pick `ParamEnv::reveal_all`, because statics don't have generics
        // and thus don't care about the parameter environment. While we could just use
        // `self.param_env`, that would mean we invoke the query to evaluate the static
        // with different parameter environments, thus causing the static to be evaluated
        // multiple times.
        let param_env = if self.tcx.is_static(gid.instance.def_id()) {
            ty::ParamEnv::reveal_all()
        } else {
            self.param_env
        };
        // We use `const_eval_raw` here, and get an unvalidated result.  That is okay:
        // Our result will later be validated anyway, and there seems no good reason
        // to have to fail early here.  This is also more consistent with
        // `Memory::get_static_alloc` which has to use `const_eval_raw` to avoid cycles.
        let val = self.tcx.const_eval_raw(param_env.and(gid))?;
        self.raw_const_to_mplace(val)
    }

    pub fn dump_place(&self, place: Place<M::PointerTag>) {
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

                match self.stack[frame].locals[local].value {
                    LocalValue::Dead => write!(msg, " is dead").unwrap(),
                    LocalValue::Uninitialized => write!(msg, " is uninitialized").unwrap(),
                    LocalValue::Live(Operand::Indirect(mplace)) => match mplace.ptr {
                        Scalar::Ptr(ptr) => {
                            write!(
                                msg,
                                " by align({}){} ref:",
                                mplace.align.bytes(),
                                match mplace.meta {
                                    MemPlaceMeta::Meta(meta) => format!(" meta({:?})", meta),
                                    MemPlaceMeta::Poison | MemPlaceMeta::None => String::new(),
                                }
                            )
                            .unwrap();
                            allocs.push(ptr.alloc_id);
                        }
                        ptr => write!(msg, " by integral ref: {:?}", ptr).unwrap(),
                    },
                    LocalValue::Live(Operand::Immediate(Immediate::Scalar(val))) => {
                        write!(msg, " {:?}", val).unwrap();
                        if let ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) = val {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                    LocalValue::Live(Operand::Immediate(Immediate::ScalarPair(val1, val2))) => {
                        write!(msg, " ({:?}, {:?})", val1, val2).unwrap();
                        if let ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) = val1 {
                            allocs.push(ptr.alloc_id);
                        }
                        if let ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) = val2 {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                }

                trace!("{}", msg);
                self.memory.dump_allocs(allocs);
            }
            Place::Ptr(mplace) => match mplace.ptr {
                Scalar::Ptr(ptr) => {
                    trace!("by align({}) ref:", mplace.align.bytes());
                    self.memory.dump_alloc(ptr.alloc_id);
                }
                ptr => trace!(" integral by ref: {:?}", ptr),
            },
        }
    }

    pub fn generate_stacktrace(&self, explicit_span: Option<Span>) -> Vec<FrameInfo<'tcx>> {
        let mut last_span = None;
        let mut frames = Vec::new();
        for frame in self.stack().iter().rev() {
            // make sure we don't emit frames that are duplicates of the previous
            if explicit_span == Some(frame.span) {
                last_span = Some(frame.span);
                continue;
            }
            if let Some(last) = last_span {
                if last == frame.span {
                    continue;
                }
            } else {
                last_span = Some(frame.span);
            }

            let lint_root = frame.current_source_info().and_then(|source_info| {
                match &frame.body.source_scopes[source_info.scope].local_data {
                    mir::ClearCrossCrate::Set(data) => Some(data.lint_root),
                    mir::ClearCrossCrate::Clear => None,
                }
            });

            frames.push(FrameInfo { call_site: frame.span, instance: frame.instance, lint_root });
        }
        trace!("generate stacktrace: {:#?}, {:?}", frames, explicit_span);
        frames
    }
}

impl<'ctx, 'mir, 'tcx, Tag, Extra> HashStable<StableHashingContext<'ctx>>
    for Frame<'mir, 'tcx, Tag, Extra>
where
    Extra: HashStable<StableHashingContext<'ctx>>,
    Tag: HashStable<StableHashingContext<'ctx>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'ctx>, hasher: &mut StableHasher) {
        self.body.hash_stable(hcx, hasher);
        self.instance.hash_stable(hcx, hasher);
        self.span.hash_stable(hcx, hasher);
        self.return_to_block.hash_stable(hcx, hasher);
        self.return_place.as_ref().map(|r| &**r).hash_stable(hcx, hasher);
        self.locals.hash_stable(hcx, hasher);
        self.block.hash_stable(hcx, hasher);
        self.stmt.hash_stable(hcx, hasher);
        self.extra.hash_stable(hcx, hasher);
    }
}
