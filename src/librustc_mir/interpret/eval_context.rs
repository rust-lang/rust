use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::mem;

use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::hir::map::definitions::DefPathData;
use rustc::mir;
use rustc::ty::layout::{
    self, Size, Align, HasDataLayout, LayoutOf, TyLayout, Primitive
};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::ty::query::TyCtxtAt;
use rustc_data_structures::fx::{FxHashSet, FxHasher};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::interpret::{
    GlobalId, Scalar, FrameInfo, AllocType,
    EvalResult, EvalErrorKind,
    ScalarMaybeUndef,
};

use syntax::source_map::{self, Span};
use syntax::ast::Mutability;

use super::{
    Value, ValTy, Operand, MemPlace, MPlaceTy, Place, PlaceExtra,
    Memory, Machine
};

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

// State of a local variable
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum LocalValue {
    Dead,
    // Mostly for convenience, we re-use the `Operand` type here.
    // This is an optimization over just always having a pointer here;
    // we can thus avoid doing an allocation when the local just stores
    // immediate values *and* never has its address taken.
    Live(Operand),
}

impl<'tcx> LocalValue {
    pub fn access(&self) -> EvalResult<'tcx, &Operand> {
        match self {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(ref val) => Ok(val),
        }
    }

    pub fn access_mut(&mut self) -> EvalResult<'tcx, &mut Operand> {
        match self {
            LocalValue::Dead => err!(DeadLocal),
            LocalValue::Live(ref mut val) => Ok(val),
        }
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

    /// Mark a storage as live, killing the previous content and returning it.
    /// Remember to deallocate that!
    pub fn storage_live(&mut self, local: mir::Local) -> EvalResult<'tcx, LocalValue> {
        trace!("{:?} is now live", local);

        let layout = self.layout_of_local(self.cur_frame(), local)?;
        let init = LocalValue::Live(self.uninit_operand(layout)?);
        // StorageLive *always* kills the value that's currently stored
        Ok(mem::replace(&mut self.frame_mut().locals[local], init))
    }

    /// Returns the old value of the local.
    /// Remember to deallocate that!
    pub fn storage_dead(&mut self, local: mir::Local) -> LocalValue {
        trace!("{:?} is now dead", local);

        mem::replace(&mut self.frame_mut().locals[local], LocalValue::Dead)
    }

    pub fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        let ptr = self.memory.allocate_bytes(s.as_bytes());
        Ok(Value::new_slice(Scalar::Ptr(ptr), s.len() as u64, self.tcx.tcx))
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

    pub fn monomorphize<T: TypeFoldable<'tcx> + Subst<'tcx>>(
        &self,
        t: T,
        substs: &'tcx Substs<'tcx>
    ) -> T {
        // miri doesn't care about lifetimes, and will choke on some crazy ones
        // let's simply get rid of them
        let substituted = t.subst(*self.tcx, substs);
        self.tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), substituted)
    }

    pub fn layout_of_local(
        &self,
        frame: usize,
        local: mir::Local
    ) -> EvalResult<'tcx, TyLayout<'tcx>> {
        let local_ty = self.stack[frame].mir.local_decls[local].ty;
        let local_ty = self.monomorphize(
            local_ty,
            self.stack[frame].instance.substs
        );
        self.layout_of(local_ty)
    }

    /// Return the size and alignment of the value at the given type.
    /// Note that the value does not matter if the type is sized. For unsized types,
    /// the value has to be a fat pointer, and we only care about the "extra" data in it.
    pub fn size_and_align_of_val(
        &self,
        val: ValTy<'tcx>,
    ) -> EvalResult<'tcx, (Size, Align)> {
        let pointee_ty = val.layout.ty.builtin_deref(true).unwrap().ty;
        let layout = self.layout_of(pointee_ty)?;
        if !layout.is_unsized() {
            Ok(layout.size_and_align())
        } else {
            match layout.ty.sty {
                ty::TyAdt(..) | ty::TyTuple(..) => {
                    // First get the size of all statically known fields.
                    // Don't use type_of::sizing_type_of because that expects t to be sized,
                    // and it also rounds up to alignment, which we want to avoid,
                    // as the unsized field's alignment could be smaller.
                    assert!(!layout.ty.is_simd());
                    debug!("DST layout: {:?}", layout);

                    let sized_size = layout.fields.offset(layout.fields.count() - 1);
                    let sized_align = layout.align;
                    debug!(
                        "DST {} statically sized prefix size: {:?} align: {:?}",
                        layout.ty,
                        sized_size,
                        sized_align
                    );

                    // Recurse to get the size of the dynamically sized field (must be
                    // the last field).
                    let field_layout = layout.field(self, layout.fields.count() - 1)?;
                    let (unsized_size, unsized_align) =
                        self.size_and_align_of_val(ValTy {
                            value: val.value,
                            layout: field_layout
                        })?;

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
                    let (_, vtable) = val.to_scalar_dyn_trait()?;
                    // the second entry in the vtable is the dynamic size of the object.
                    self.read_size_and_align_from_vtable(vtable)
                }

                ty::TySlice(_) | ty::TyStr => {
                    let (elem_size, align) = layout.field(self, 0)?.size_and_align();
                    let (_, len) = val.to_scalar_slice(self)?;
                    Ok((elem_size * len, align))
                }

                _ => bug!("size_of_val::<{:?}>", layout.ty),
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
            // We put some marker value into the locals that we later want to initialize.
            // This can be anything except for LocalValue::Dead -- because *that* is the
            // value we use for things that we know are initially dead.
            let dummy =
                LocalValue::Live(Operand::Immediate(Value::Scalar(ScalarMaybeUndef::Undef)));
            self.frame_mut().locals = IndexVec::from_elem(dummy, &mir.local_decls);
            // Now mark those locals as dead that we do not want to initialize
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
                                StorageDead(local) => {
                                    // Worst case we are overwriting a dummy, no deallocation needed
                                    self.storage_dead(local);
                                }
                                _ => {}
                            }
                        }
                    }
                },
            }
            // Finally, properly initialize all those that still have the dummy value
            for local in mir.local_decls.indices() {
                if self.frame().locals[local] == dummy {
                    self.storage_live(local)?;
                }
            }
        }

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
        match frame.return_to_block {
            StackPopCleanup::MarkStatic(mutable) => {
                if let Place::Ptr(MemPlace { ptr, .. }) = frame.return_place {
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

    crate fn deallocate_local(&mut self, local: LocalValue) -> EvalResult<'tcx> {
        // FIXME: should we tell the user that there was a local which was never written to?
        if let LocalValue::Live(Operand::Indirect(MemPlace { ptr, .. })) = local {
            trace!("deallocating local");
            let ptr = ptr.to_ptr()?;
            self.memory.dump_alloc(ptr.alloc_id);
            self.memory.deallocate_local(ptr)?;
        };
        Ok(())
    }

    pub fn const_eval(&self, gid: GlobalId<'tcx>) -> EvalResult<'tcx, &'tcx ty::Const<'tcx>> {
        let param_env = if self.tcx.is_static(gid.instance.def_id()).is_some() {
            ty::ParamEnv::reveal_all()
        } else {
            self.param_env
        };
        self.tcx.const_eval(param_env.and(gid)).map_err(|err| EvalErrorKind::ReferencedConstant(err).into())
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
    pub fn validate_mplace(
        &self,
        dest: MPlaceTy<'tcx>,
        path: String,
        seen: &mut FxHashSet<(MPlaceTy<'tcx>)>,
        todo: &mut Vec<(MPlaceTy<'tcx>, String)>,
    ) -> EvalResult<'tcx> {
        self.memory.dump_alloc(dest.to_ptr()?.alloc_id);
        trace!("validate_mplace: {:?}, {:#?}", *dest, dest.layout);

        // Find the right variant
        let (variant, dest) = match dest.layout.variants {
            layout::Variants::NicheFilling { niche: ref tag, .. } |
            layout::Variants::Tagged { ref tag, .. } => {
                let size = tag.value.size(self);
                // we first read the tag value as scalar, to be able to validate it
                let tag_mplace = self.mplace_field(dest, 0)?;
                let tag_value = self.read_scalar(tag_mplace.into())?;
                let path = format!("{}.TAG", path);
                self.validate_scalar(
                    tag_value, size, tag, &path, tag_mplace.layout.ty
                )?;
                // then we read it again to get the index, to continue
                let variant = self.read_discriminant_as_variant_index(dest.into())?;
                let dest = self.mplace_downcast(dest, variant)?;
                trace!("variant layout: {:#?}", dest.layout);
                (variant, dest)
            },
            layout::Variants::Single { index } => {
                (index, dest)
            }
        };

        // Validate all fields
        match dest.layout.fields {
            // primitives are unions with zero fields
            layout::FieldPlacement::Union(0) => {
                match dest.layout.abi {
                    // nothing to do, whatever the pointer points to, it is never going to be read
                    layout::Abi::Uninhabited => validation_failure!("a value of an uninhabited type", path),
                    // check that the scalar is a valid pointer or that its bit range matches the
                    // expectation.
                    layout::Abi::Scalar(ref scalar_layout) => {
                        let size = scalar_layout.value.size(self);
                        let value = self.read_value(dest.into())?;
                        let scalar = value.to_scalar_or_undef();
                        self.validate_scalar(scalar, size, scalar_layout, &path, dest.layout.ty)?;
                        if scalar_layout.value == Primitive::Pointer {
                            // ignore integer pointers, we can't reason about the final hardware
                            if let Scalar::Ptr(ptr) = scalar.not_undef()? {
                                let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                                if let Some(AllocType::Static(did)) = alloc_kind {
                                    // statics from other crates are already checked
                                    // extern statics should not be validated as they have no body
                                    if !did.is_local() || self.tcx.is_foreign_item(did) {
                                        return Ok(());
                                    }
                                }
                                if value.layout.ty.builtin_deref(false).is_some() {
                                    trace!("Recursing below ptr {:#?}", value);
                                    let ptr_place = self.ref_to_mplace(value)?;
                                    // we have not encountered this pointer+layout combination before
                                    if seen.insert(ptr_place) {
                                        todo.push((ptr_place, format!("(*{})", path)))
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                    _ => bug!("bad abi for FieldPlacement::Union(0): {:#?}", dest.layout.abi),
                }
            }
            layout::FieldPlacement::Union(_) => {
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
                Ok(())
            },
            layout::FieldPlacement::Array { count, .. } => {
                for i in 0..count {
                    let mut path = path.clone();
                    self.dump_field_name(&mut path, dest.layout.ty, i as usize, variant).unwrap();
                    let field = self.mplace_field(dest, i)?;
                    self.validate_mplace(field, path, seen, todo)?;
                }
                Ok(())
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                // fat pointers need special treatment
                match dest.layout.ty.builtin_deref(false).map(|tam| &tam.ty.sty) {
                    | Some(ty::TyStr)
                    | Some(ty::TySlice(_)) => {
                        // check the length (for nicer error messages)
                        let len_mplace = self.mplace_field(dest, 1)?;
                        let len = self.read_scalar(len_mplace.into())?;
                        let len = match len.to_bits(len_mplace.layout.size) {
                            Err(_) => return validation_failure!("length is not a valid integer", path),
                            Ok(len) => len as u64,
                        };
                        // get the fat ptr, and recursively check it
                        let ptr = self.ref_to_mplace(self.read_value(dest.into())?)?;
                        assert_eq!(ptr.extra, PlaceExtra::Length(len));
                        let unpacked_ptr = self.unpack_unsized_mplace(ptr)?;
                        if seen.insert(unpacked_ptr) {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, 0, 0).unwrap();
                            todo.push((unpacked_ptr, path))
                        }
                    },
                    Some(ty::TyDynamic(..)) => {
                        // check the vtable (for nicer error messages)
                        let vtable = self.read_scalar(self.mplace_field(dest, 1)?.into())?;
                        let vtable = match vtable.to_ptr() {
                            Err(_) => return validation_failure!("vtable address is not a pointer", path),
                            Ok(vtable) => vtable,
                        };
                        // get the fat ptr, and recursively check it
                        let ptr = self.ref_to_mplace(self.read_value(dest.into())?)?;
                        assert_eq!(ptr.extra, PlaceExtra::Vtable(vtable));
                        let unpacked_ptr = self.unpack_unsized_mplace(ptr)?;
                        if seen.insert(unpacked_ptr) {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, 0, 0).unwrap();
                            todo.push((unpacked_ptr, path))
                        }
                        // FIXME: More checks for the vtable... making sure it is exactly
                        // the one one would expect for this type.
                    },
                    Some(ty) =>
                        bug!("Unexpected fat pointer target type {:?}", ty),
                    None => {
                        // Not a pointer, perform regular aggregate handling below
                        for i in 0..offsets.len() {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, i, variant).unwrap();
                            let field = self.mplace_field(dest, i as u64)?;
                            self.validate_mplace(field, path, seen, todo)?;
                        }
                        // FIXME: For a TyStr, check that this is valid UTF-8.
                    },
                }

                Ok(())
            }
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

    pub fn dump_place(&self, place: Place) {
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
                    Ok(Operand::Indirect(mplace)) => {
                        let (ptr, align) = mplace.to_scalar_ptr_align();
                        match ptr {
                            Scalar::Ptr(ptr) => {
                                write!(msg, " by align({}) ref:", align.abi()).unwrap();
                                allocs.push(ptr.alloc_id);
                            }
                            ptr => write!(msg, " by integral ref: {:?}", ptr).unwrap(),
                        }
                    }
                    Ok(Operand::Immediate(Value::Scalar(val))) => {
                        write!(msg, " {:?}", val).unwrap();
                        if let ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) = val {
                            allocs.push(ptr.alloc_id);
                        }
                    }
                    Ok(Operand::Immediate(Value::ScalarPair(val1, val2))) => {
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
            Place::Ptr(mplace) => {
                match mplace.ptr {
                    Scalar::Ptr(ptr) => {
                        trace!("by align({}) ref:", mplace.align.abi());
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
        assert!(ty.abi.is_signed());
        super::sign_extend(value, ty.size)
    }

    pub fn truncate(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        super::truncate(value, ty.size)
    }

    fn dump_field_name(&self, s: &mut String, ty: Ty<'tcx>, i: usize, variant: usize) -> ::std::fmt::Result {
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
                bug!("dump_field_name: unexpected type `{}`", ty)
            }
        }
    }
}

pub fn sign_extend(value: u128, size: Size) -> u128 {
    let size = size.bits();
    // sign extend
    let shift = 128 - size;
    // shift the unsigned value to the left
    // and back to the right as signed (essentially fills with FF on the left)
    (((value << shift) as i128) >> shift) as u128
}

pub fn truncate(value: u128, size: Size) -> u128 {
    let size = size.bits();
    let shift = 128 - size;
    // truncate (shift left to drop out leftover values, shift right to fill with zeroes)
    (value << shift) >> shift
}
