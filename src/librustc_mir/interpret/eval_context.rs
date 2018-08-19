use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::mem;

use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::hir::map::definitions::DefPathData;
use rustc::mir;
use rustc::ty::layout::{self, Size, Align, HasDataLayout, IntegerExt, LayoutOf, TyLayout, Primitive};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, TypeAndMut};
use rustc::ty::query::TyCtxtAt;
use rustc_data_structures::fx::{FxHashSet, FxHasher};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::mir::interpret::{
    GlobalId, Value, Scalar, FrameInfo, AllocType,
    EvalResult, EvalErrorKind, Pointer, ConstValue,
    ScalarMaybeUndef,
};

use syntax::source_map::{self, Span};
use syntax::ast::Mutability;

use super::{Place, PlaceExtra, Memory,
            HasMemory, MemoryKind,
            Machine};

macro_rules! validation_failure{
    ($what:expr, $where:expr, $details:expr) => {{
        let where_ = if $where.is_empty() {
            String::new()
        } else {
            format!(" at {}", $where)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}, but expected {}",
            $what, where_, $details,
        )))
    }};
    ($what:expr, $where:expr) => {{
        let where_ = if $where.is_empty() {
            String::new()
        } else {
            format!(" at {}", $where)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}",
            $what, where_,
        )))
    }};
}

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

    /// When this value is negative, it indicates the number of interpreter
    /// steps *until* the loop detector is enabled. When it is positive, it is
    /// the number of steps after the detector has been enabled modulo the loop
    /// detector period.
    pub(crate) steps_since_detector_enabled: isize,

    pub(crate) loop_detector: InfiniteLoopDetector<'a, 'mir, 'tcx, M>,
}

/// A stack frame.
#[derive(Clone)]
pub struct Frame<'mir, 'tcx: 'mir> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub mir: &'mir mir::Mir<'tcx>,

    /// The def_id and substs of the current function
    pub instance: ty::Instance<'tcx>,

    /// The span of the call site.
    pub span: source_map::Span,

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
    pub locals: IndexVec<mir::Local, LocalValue>,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////
    /// The block that is currently executed (or will be executed after the above call stacks
    /// return).
    pub block: mir::BasicBlock,

    /// The index of the currently evaluated statement.
    pub stmt: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum LocalValue {
    Dead,
    Live(Value),
}

impl LocalValue {
    pub fn access(self) -> EvalResult<'static, Value> {
        match self {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(val) => Ok(val),
        }
    }
}

impl<'mir, 'tcx: 'mir> Eq for Frame<'mir, 'tcx> {}

impl<'mir, 'tcx: 'mir> PartialEq for Frame<'mir, 'tcx> {
    fn eq(&self, other: &Self) -> bool {
        let Frame {
            mir: _,
            instance,
            span: _,
            return_to_block,
            return_place,
            locals,
            block,
            stmt,
        } = self;

        // Some of these are constant during evaluation, but are included
        // anyways for correctness.
        *instance == other.instance
            && *return_to_block == other.return_to_block
            && *return_place == other.return_place
            && *locals == other.locals
            && *block == other.block
            && *stmt == other.stmt
    }
}

impl<'mir, 'tcx: 'mir> Hash for Frame<'mir, 'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Frame {
            mir: _,
            instance,
            span: _,
            return_to_block,
            return_place,
            locals,
            block,
            stmt,
        } = self;

        instance.hash(state);
        return_to_block.hash(state);
        return_place.hash(state);
        locals.hash(state);
        block.hash(state);
        stmt.hash(state);
    }
}

/// The virtual machine state during const-evaluation at a given point in time.
type EvalSnapshot<'a, 'mir, 'tcx, M>
    = (M, Vec<Frame<'mir, 'tcx>>, Memory<'a, 'mir, 'tcx, M>);

pub(crate) struct InfiniteLoopDetector<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    /// The set of all `EvalSnapshot` *hashes* observed by this detector.
    ///
    /// When a collision occurs in this table, we store the full snapshot in
    /// `snapshots`.
    hashes: FxHashSet<u64>,

    /// The set of all `EvalSnapshot`s observed by this detector.
    ///
    /// An `EvalSnapshot` will only be fully cloned once it has caused a
    /// collision in `hashes`. As a result, the detector must observe at least
    /// *two* full cycles of an infinite loop before it triggers.
    snapshots: FxHashSet<EvalSnapshot<'a, 'mir, 'tcx, M>>,
}

impl<'a, 'mir, 'tcx, M> Default for InfiniteLoopDetector<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    fn default() -> Self {
        InfiniteLoopDetector {
            hashes: FxHashSet::default(),
            snapshots: FxHashSet::default(),
        }
    }
}

impl<'a, 'mir, 'tcx, M> InfiniteLoopDetector<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    /// Returns `true` if the loop detector has not yet observed a snapshot.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub fn observe_and_analyze(
        &mut self,
        machine: &M,
        stack: &Vec<Frame<'mir, 'tcx>>,
        memory: &Memory<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, ()> {
        let snapshot = (machine, stack, memory);

        let mut fx = FxHasher::default();
        snapshot.hash(&mut fx);
        let hash = fx.finish();

        if self.hashes.insert(hash) {
            // No collision
            return Ok(())
        }

        if self.snapshots.insert((machine.clone(), stack.clone(), memory.clone())) {
            // Spurious collision or first cycle
            return Ok(())
        }

        // Second cycle
        Err(EvalErrorKind::InfiniteLoop.into())
    }
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

const STEPS_UNTIL_DETECTOR_ENABLED: isize = 1_000_000;

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
            loop_detector: Default::default(),
            steps_since_detector_enabled: -STEPS_UNTIL_DETECTOR_ENABLED,
        }
    }

    pub(crate) fn with_fresh_body<F: FnOnce(&mut Self) -> R, R>(&mut self, f: F) -> R {
        let stack = mem::replace(&mut self.stack, Vec::new());
        let steps = mem::replace(&mut self.steps_since_detector_enabled, -STEPS_UNTIL_DETECTOR_ENABLED);
        let r = f(self);
        self.stack = stack;
        self.steps_since_detector_enabled = steps;
        r
    }

    pub fn alloc_ptr(&mut self, layout: TyLayout<'tcx>) -> EvalResult<'tcx, Pointer> {
        assert!(!layout.is_unsized(), "cannot alloc memory for unsized type");

        self.memory.allocate(layout.size, layout.align, MemoryKind::Stack)
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

    pub fn const_to_value(
        &mut self,
        val: ConstValue<'tcx>,
    ) -> EvalResult<'tcx, Value> {
        match val {
            ConstValue::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                self.read_global_as_value(GlobalId {
                    instance,
                    promoted: None,
                })
            }
            ConstValue::ByRef(alloc, offset) => {
                // FIXME: Allocate new AllocId for all constants inside
                let id = self.memory.allocate_value(alloc.clone(), MemoryKind::Stack)?;
                Ok(Value::ByRef(Pointer::new(id, offset).into(), alloc.align))
            },
            ConstValue::ScalarPair(a, b) => Ok(Value::ScalarPair(a.into(), b.into())),
            ConstValue::Scalar(val) => Ok(Value::Scalar(val.into())),
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
        ).ok_or_else(|| EvalErrorKind::TooGeneric.into())
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
        &self,
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
                    let field_ty = layout.field(self, layout.fields.count() - 1)?.ty;
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
                    let (elem_size, align) = layout.field(self, 0)?.size_and_align();
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
        span: source_map::Span,
        mir: &'mir mir::Mir<'tcx>,
        return_place: Place,
        return_to_block: StackPopCleanup,
    ) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation += 1;

        // first push a stack frame so we have access to the local substs
        self.stack.push(Frame {
            mir,
            block: mir::START_BLOCK,
            return_to_block,
            return_place,
            // empty local array, we fill it in below, after we are inside the stack frame and
            // all methods actually know about the frame
            locals: IndexVec::new(),
            span,
            instance,
            stmt: 0,
        });

        // don't allocate at all for trivial constants
        if mir.local_decls.len() > 1 {
            let mut locals = IndexVec::from_elem(LocalValue::Dead, &mir.local_decls);
            for (local, decl) in locals.iter_mut().zip(mir.local_decls.iter()) {
                *local = LocalValue::Live(self.init_value(decl.ty)?);
            }
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
                                StorageDead(local) => locals[local] = LocalValue::Dead,
                                _ => {}
                            }
                        }
                    }
                },
            }
            self.frame_mut().locals = locals;
        }

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
                        ptr.unwrap_or_err()?.to_ptr()?.alloc_id,
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

    pub fn deallocate_local(&mut self, local: LocalValue) -> EvalResult<'tcx> {
        // FIXME: should we tell the user that there was a local which was never written to?
        if let LocalValue::Live(Value::ByRef(ptr, _align)) = local {
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
        let dest_layout = self.layout_of(dest_ty)?;

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
                let val = self.unary_op(un_op, val, dest_layout)?;
                self.write_scalar(
                    dest,
                    val,
                    dest_ty,
                )?;
            }

            Aggregate(ref kind, ref operands) => {
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

                if length > 0 {
                    let dest = dest.unwrap_or_err()?;
                    //write the first value
                    self.write_value_to_ptr(value, dest, dest_align, elem_ty)?;

                    if length > 1 {
                        let rest = dest.ptr_offset(elem_size * 1 as u64, &self)?;
                        self.memory.copy_repeatedly(dest, dest_align, rest, dest_align, elem_size, length - 1, false)?;
                    }
                }
            }

            Len(ref place) => {
                // FIXME(CTFE): don't allow computing the length of arrays in const eval
                let src = self.eval_place(place)?;
                let ty = self.place_ty(place);
                let (_, len) = src.elem_ty_and_len(ty, self.tcx.tcx);
                let size = self.memory.pointer_size().bytes() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: len as u128,
                        size,
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
                    PlaceExtra::None => Value::Scalar(ptr),
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
                let size = self.memory.pointer_size().bytes() as u8;
                self.write_scalar(
                    dest,
                    Scalar::Bits {
                        bits: layout.size.bytes() as u128,
                        size,
                    },
                    dest_ty,
                )?;
            }

            Cast(kind, ref operand, cast_ty) => {
                debug_assert_eq!(self.monomorphize(cast_ty, self.substs()), dest_ty);
                let src = self.eval_operand(operand)?;
                self.cast(src, kind, dest_ty, dest)?;
            }

            Discriminant(ref place) => {
                let ty = self.place_ty(place);
                let layout = self.layout_of(ty)?;
                let place = self.eval_place(place)?;
                let discr_val = self.read_discriminant_value(place, layout)?;
                let size = self.layout_of(dest_ty).unwrap().size.bytes() as u8;
                self.write_scalar(dest, Scalar::Bits {
                    bits: discr_val,
                    size,
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
                let value = self.const_to_value(constant.literal.val)?;

                Ok(ValTy {
                    value,
                    ty,
                })
            }
        }
    }

    /// reads a tag and produces the corresponding variant index
    pub fn read_discriminant_as_variant_index(
        &self,
        place: Place,
        layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, usize> {
        match layout.variants {
            ty::layout::Variants::Single { index } => Ok(index),
            ty::layout::Variants::Tagged { .. } => {
                let discr_val = self.read_discriminant_value(place, layout)?;
                layout
                    .ty
                    .ty_adt_def()
                    .expect("tagged layout for non adt")
                    .discriminants(self.tcx.tcx)
                    .position(|var| var.val == discr_val)
                    .ok_or_else(|| EvalErrorKind::InvalidDiscriminant.into())
            }
            ty::layout::Variants::NicheFilling { .. } => {
                let discr_val = self.read_discriminant_value(place, layout)?;
                assert_eq!(discr_val as usize as u128, discr_val);
                Ok(discr_val as usize)
            },
        }
    }

    pub fn read_discriminant_value(
        &self,
        place: Place,
        layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, u128> {
        trace!("read_discriminant_value {:#?}", layout);
        if layout.abi == layout::Abi::Uninhabited {
            return Ok(0);
        }

        match layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = layout.ty.ty_adt_def().map_or(
                    index as u128,
                    |def| def.discriminant_for_variant(*self.tcx, index).val);
                return Ok(discr_val);
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }
        let discr_place_val = self.read_place(place)?;
        let (discr_val, discr) = self.read_field(discr_place_val, None, mir::Field::new(0), layout)?;
        trace!("discr value: {:?}, {:?}", discr_val, discr);
        let raw_discr = self.value_to_scalar(ValTy {
            value: discr_val,
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
                    let discr_ty = layout
                        .ty
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
                    Scalar::Bits { bits: raw_discr, size } => {
                        assert_eq!(size as u64, discr.size.bytes());
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
                let size = tag.value.size(self.tcx.tcx);
                let shift = 128 - size.bits();
                let discr_val = (discr_val << shift) >> shift;

                let (discr_dest, tag) = self.place_field(dest, mir::Field::new(0), layout)?;
                self.write_scalar(discr_dest, Scalar::Bits {
                    bits: discr_val,
                    size: size.bytes() as u8,
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
                        size: niche.size.bytes() as u8,
                    }, niche.ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn read_global_as_value(&mut self, gid: GlobalId<'tcx>) -> EvalResult<'tcx, Value> {
        let cv = self.const_eval(gid)?;
        self.const_to_value(cv.val)
    }

    pub fn const_eval(&self, gid: GlobalId<'tcx>) -> EvalResult<'tcx, &'tcx ty::Const<'tcx>> {
        let param_env = if self.tcx.is_static(gid.instance.def_id()).is_some() {
            ty::ParamEnv::reveal_all()
        } else {
            self.param_env
        };
        self.tcx.const_eval(param_env.and(gid)).map_err(|err| EvalErrorKind::ReferencedConstant(err).into())
    }

    pub fn allocate_place_for_value(
        &mut self,
        value: Value,
        layout: TyLayout<'tcx>,
        variant: Option<usize>,
    ) -> EvalResult<'tcx, Place> {
        let (ptr, align) = match value {
            Value::ByRef(ptr, align) => (ptr, align),
            Value::ScalarPair(..) | Value::Scalar(_) => {
                let ptr = self.alloc_ptr(layout)?.into();
                self.write_value_to_ptr(value, ptr, layout.align, layout.ty)?;
                (ptr, layout.align)
            },
        };
        Ok(Place::Ptr {
            ptr: ptr.into(),
            align,
            extra: variant.map_or(PlaceExtra::None, PlaceExtra::DowncastVariant),
        })
    }

    pub fn force_allocation(&mut self, place: Place) -> EvalResult<'tcx, Place> {
        let new_place = match place {
            Place::Local { frame, local } => {
                match self.stack[frame].locals[local].access()? {
                    Value::ByRef(ptr, align) => {
                        Place::Ptr {
                            ptr: ptr.into(),
                            align,
                            extra: PlaceExtra::None,
                        }
                    }
                    val => {
                        let ty = self.stack[frame].mir.local_decls[local].ty;
                        let ty = self.monomorphize(ty, self.stack[frame].instance.substs);
                        let layout = self.layout_of(ty)?;
                        let ptr = self.alloc_ptr(layout)?;
                        self.stack[frame].locals[local] =
                            LocalValue::Live(Value::ByRef(ptr.into(), layout.align)); // it stays live

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

            Value::Scalar(scalar) => scalar.unwrap_or_err(),

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
        val: impl Into<ScalarMaybeUndef>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let valty = ValTy {
            value: Value::Scalar(val.into()),
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
                self.write_value_to_ptr(src_val, ptr.unwrap_or_err()?, align, dest_ty)
            }

            Place::Local { frame, local } => {
                let old_val = self.stack[frame].locals[local].access()?;
                self.write_value_possibly_by_val(
                    src_val,
                    |this, val| this.stack[frame].set_local(local, val),
                    old_val,
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
        // FIXME: this should be a layout check, not underlying value
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
                let layout = self.layout_of(dest_ty)?;
                let dest_ptr = self.alloc_ptr(layout)?.into();
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
                    _ => false,
                };
                self.memory.write_scalar(dest, dest_align, scalar, layout.size, layout.align, signed)
            }
            Value::ScalarPair(a_val, b_val) => {
                trace!("write_value_to_ptr valpair: {:#?}", layout);
                let (a, b) = match layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (&a.value, &b.value),
                    _ => bug!("write_value_to_ptr: invalid ScalarPair layout: {:#?}", layout)
                };
                let (a_size, b_size) = (a.size(&self), b.size(&self));
                let (a_align, b_align) = (a.align(&self), b.align(&self));
                let a_ptr = dest;
                let b_offset = a_size.abi_align(b_align);
                let b_ptr = dest.ptr_offset(b_offset, &self)?.into();
                // TODO: What about signedess?
                self.memory.write_scalar(a_ptr, dest_align, a_val, a_size, a_align, false)?;
                self.memory.write_scalar(b_ptr, dest_align, b_val, b_size, b_align, false)
            }
        }
    }

    pub fn read_value(&self, ptr: Scalar, align: Align, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, align, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    fn validate_scalar(
        &self,
        value: ScalarMaybeUndef,
        size: Size,
        scalar: &layout::Scalar,
        path: &str,
        ty: Ty,
    ) -> EvalResult<'tcx> {
        trace!("validate scalar: {:#?}, {:#?}, {:#?}, {}", value, size, scalar, ty);
        let (lo, hi) = scalar.valid_range.clone().into_inner();

        let value = match value {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return validation_failure!("undefined bytes", path),
        };

        let bits = match value {
            Scalar::Bits { bits, size: value_size } => {
                assert_eq!(value_size as u64, size.bytes());
                bits
            },
            Scalar::Ptr(_) => {
                let ptr_size = self.memory.pointer_size();
                let ptr_max = u128::max_value() >> (128 - ptr_size.bits());
                return if lo > hi {
                    if lo - hi == 1 {
                        // no gap, all values are ok
                        Ok(())
                    } else if hi < ptr_max || lo > 1 {
                        let max = u128::max_value() >> (128 - size.bits());
                        validation_failure!(
                            "pointer",
                            path,
                            format!("something in the range {:?} or {:?}", 0..=lo, hi..=max)
                        )
                    } else {
                        Ok(())
                    }
                } else if hi < ptr_max || lo > 1 {
                    validation_failure!(
                        "pointer",
                        path,
                        format!("something in the range {:?}", scalar.valid_range)
                    )
                } else {
                    Ok(())
                };
            },
        };

        // char gets a special treatment, because its number space is not contiguous so `TyLayout`
        // has no special checks for chars
        match ty.sty {
            ty::TyChar => {
                debug_assert_eq!(size.bytes(), 4);
                if ::std::char::from_u32(bits as u32).is_none() {
                    return err!(InvalidChar(bits));
                }
            }
            _ => {},
        }

        use std::ops::RangeInclusive;
        let in_range = |bound: RangeInclusive<u128>| bound.contains(&bits);
        if lo > hi {
            if in_range(0..=hi) || in_range(lo..=u128::max_value()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?} or {:?}", ..=hi, lo..)
                )
            }
        } else {
            if in_range(scalar.valid_range.clone()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?}", scalar.valid_range)
                )
            }
        }
    }

    /// This function checks the memory where `ptr` points to.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    pub fn validate_ptr_target(
        &self,
        ptr: Pointer,
        ptr_align: Align,
        mut layout: TyLayout<'tcx>,
        path: String,
        seen: &mut FxHashSet<(Pointer, Ty<'tcx>)>,
        todo: &mut Vec<(Pointer, Ty<'tcx>, String)>,
    ) -> EvalResult<'tcx> {
        self.memory.dump_alloc(ptr.alloc_id);
        trace!("validate_ptr_target: {:?}, {:#?}", ptr, layout);

        let variant;
        match layout.variants {
            layout::Variants::NicheFilling { niche: ref tag, .. } |
            layout::Variants::Tagged { ref tag, .. } => {
                let size = tag.value.size(self);
                let (tag_value, tag_layout) = self.read_field(
                    Value::ByRef(ptr.into(), ptr_align),
                    None,
                    mir::Field::new(0),
                    layout,
                )?;
                let tag_value = match self.follow_by_ref_value(tag_value, tag_layout.ty)? {
                    Value::Scalar(val) => val,
                    _ => bug!("tag must be scalar"),
                };
                let path = format!("{}.TAG", path);
                self.validate_scalar(tag_value, size, tag, &path, tag_layout.ty)?;
                let variant_index = self.read_discriminant_as_variant_index(
                    Place::from_ptr(ptr, ptr_align),
                    layout,
                )?;
                variant = variant_index;
                layout = layout.for_variant(self, variant_index);
                trace!("variant layout: {:#?}", layout);
            },
            layout::Variants::Single { index } => variant = index,
        }
        match layout.fields {
            // primitives are unions with zero fields
            layout::FieldPlacement::Union(0) => {
                match layout.abi {
                    // nothing to do, whatever the pointer points to, it is never going to be read
                    layout::Abi::Uninhabited => validation_failure!("a value of an uninhabited type", path),
                    // check that the scalar is a valid pointer or that its bit range matches the
                    // expectation.
                    layout::Abi::Scalar(ref scalar) => {
                        let size = scalar.value.size(self);
                        let value = self.memory.read_scalar(ptr, ptr_align, size)?;
                        self.validate_scalar(value, size, scalar, &path, layout.ty)?;
                        if scalar.value == Primitive::Pointer {
                            // ignore integer pointers, we can't reason about the final hardware
                            if let Scalar::Ptr(ptr) = value.unwrap_or_err()? {
                                let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                                if let Some(AllocType::Static(did)) = alloc_kind {
                                    // statics from other crates are already checked
                                    // extern statics should not be validated as they have no body
                                    if !did.is_local() || self.tcx.is_foreign_item(did) {
                                        return Ok(());
                                    }
                                }
                                if let Some(tam) = layout.ty.builtin_deref(false) {
                                    // we have not encountered this pointer+layout combination before
                                    if seen.insert((ptr, tam.ty)) {
                                        todo.push((ptr, tam.ty, format!("(*{})", path)))
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                    _ => bug!("bad abi for FieldPlacement::Union(0): {:#?}", layout.abi),
                }
            }
            layout::FieldPlacement::Union(_) => {
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
                Ok(())
            },
            layout::FieldPlacement::Array { stride, count } => {
                let elem_layout = layout.field(self, 0)?;
                for i in 0..count {
                    let mut path = path.clone();
                    self.write_field_name(&mut path, layout.ty, i as usize, variant).unwrap();
                    self.validate_ptr_target(ptr.offset(stride * i, self)?, ptr_align, elem_layout, path, seen, todo)?;
                }
                Ok(())
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {

                // check length field and vtable field
                match layout.ty.builtin_deref(false).map(|tam| &tam.ty.sty) {
                    | Some(ty::TyStr)
                    | Some(ty::TySlice(_)) => {
                        let (len, len_layout) = self.read_field(
                            Value::ByRef(ptr.into(), ptr_align),
                            None,
                            mir::Field::new(1),
                            layout,
                        )?;
                        let len = self.value_to_scalar(ValTy { value: len, ty: len_layout.ty })?;
                        if len.to_bits(len_layout.size).is_err() {
                            return validation_failure!("length is not a valid integer", path);
                        }
                    },
                    Some(ty::TyDynamic(..)) => {
                        let (vtable, vtable_layout) = self.read_field(
                            Value::ByRef(ptr.into(), ptr_align),
                            None,
                            mir::Field::new(1),
                            layout,
                        )?;
                        let vtable = self.value_to_scalar(ValTy { value: vtable, ty: vtable_layout.ty })?;
                        if vtable.to_ptr().is_err() {
                            return validation_failure!("vtable address is not a pointer", path);
                        }
                    }
                    _ => {},
                }
                for (i, &offset) in offsets.iter().enumerate() {
                    let field_layout = layout.field(self, i)?;
                    let mut path = path.clone();
                    self.write_field_name(&mut path, layout.ty, i, variant).unwrap();
                    self.validate_ptr_target(ptr.offset(offset, self)?, ptr_align, field_layout, path, seen, todo)?;
                }
                Ok(())
            }
        }
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
            return Ok(Some(Value::Scalar(ScalarMaybeUndef::Scalar(Scalar::Bits { bits: 0, size: 0 }))));
        }

        let ptr = ptr.to_ptr()?;

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

    crate fn unsize_into(
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
                            let src_place = Place::from_scalar_ptr(ptr.into(), align);
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

                match self.stack[frame].locals[local].access() {
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
                        if let ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) = val {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                    Ok(Value::ScalarPair(val1, val2)) => {
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
            Place::Ptr { ptr, align, .. } => {
                match ptr {
                    ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) => {
                        trace!("by align({}) ref:", align.abi());
                        self.memory.dump_alloc(ptr.alloc_id);
                    }
                    ptr => trace!(" integral by ref: {:?}", ptr),
                }
            }
        }
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

    pub fn sign_extend(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        super::sign_extend(value, ty)
    }

    pub fn truncate(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        super::truncate(value, ty)
    }

    fn write_field_name(&self, s: &mut String, ty: Ty<'tcx>, i: usize, variant: usize) -> ::std::fmt::Result {
        match ty.sty {
            ty::TyBool |
            ty::TyChar |
            ty::TyInt(_) |
            ty::TyUint(_) |
            ty::TyFloat(_) |
            ty::TyFnPtr(_) |
            ty::TyNever |
            ty::TyFnDef(..) |
            ty::TyGeneratorWitness(..) |
            ty::TyForeign(..) |
            ty::TyDynamic(..) => {
                bug!("field_name({:?}): not applicable", ty)
            }

            // Potentially-fat pointers.
            ty::TyRef(_, pointee, _) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                assert!(i < 2);

                // Reuse the fat *T type as its own thin pointer data field.
                // This provides information about e.g. DST struct pointees
                // (which may have no non-DST form), and will work as long
                // as the `Abi` or `FieldPlacement` is checked by users.
                if i == 0 {
                    return write!(s, ".data_ptr");
                }

                match self.tcx.struct_tail(pointee).sty {
                    ty::TySlice(_) |
                    ty::TyStr => write!(s, ".len"),
                    ty::TyDynamic(..) => write!(s, ".vtable_ptr"),
                    _ => bug!("field_name({:?}): not applicable", ty)
                }
            }

            // Arrays and slices.
            ty::TyArray(_, _) |
            ty::TySlice(_) |
            ty::TyStr => write!(s, "[{}]", i),

            // generators and closures.
            ty::TyClosure(def_id, _) | ty::TyGenerator(def_id, _, _) => {
                let node_id = self.tcx.hir.as_local_node_id(def_id).unwrap();
                let freevar = self.tcx.with_freevars(node_id, |fv| fv[i]);
                write!(s, ".upvar({})", self.tcx.hir.name(freevar.var_id()))
            }

            ty::TyTuple(_) => write!(s, ".{}", i),

            // enums
            ty::TyAdt(def, ..) if def.is_enum() => {
                let variant = &def.variants[variant];
                write!(s, ".{}::{}", variant.name, variant.fields[i].ident)
            }

            // other ADTs.
            ty::TyAdt(def, _) => write!(s, ".{}", def.non_enum_variant().fields[i].ident),

            ty::TyProjection(_) | ty::TyAnon(..) | ty::TyParam(_) |
            ty::TyInfer(_) | ty::TyError => {
                bug!("write_field_name: unexpected type `{}`", ty)
            }
        }
    }

    pub fn storage_live(&mut self, local: mir::Local) -> EvalResult<'tcx, LocalValue> {
        trace!("{:?} is now live", local);

        let ty = self.frame().mir.local_decls[local].ty;
        let init = self.init_value(ty)?;
        // StorageLive *always* kills the value that's currently stored
        Ok(mem::replace(&mut self.frame_mut().locals[local], LocalValue::Live(init)))
    }

    fn init_value(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        let ty = self.monomorphize(ty, self.substs());
        let layout = self.layout_of(ty)?;
        Ok(match layout.abi {
            layout::Abi::Scalar(..) => Value::Scalar(ScalarMaybeUndef::Undef),
            layout::Abi::ScalarPair(..) => Value::ScalarPair(
                ScalarMaybeUndef::Undef,
                ScalarMaybeUndef::Undef,
            ),
            _ => Value::ByRef(self.alloc_ptr(layout)?.into(), layout.align),
        })
    }
}

impl<'mir, 'tcx> Frame<'mir, 'tcx> {
    fn set_local(&mut self, local: mir::Local, value: Value) -> EvalResult<'tcx> {
        match self.locals[local] {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(ref mut local) => {
                *local = value;
                Ok(())
            }
        }
    }

    /// Returns the old value of the local
    pub fn storage_dead(&mut self, local: mir::Local) -> LocalValue {
        trace!("{:?} is now dead", local);

        mem::replace(&mut self.locals[local], LocalValue::Dead)
    }
}
