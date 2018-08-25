// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::mem;

use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::hir::map::definitions::DefPathData;
use rustc::mir;
use rustc::ty::layout::{
    self, Size, Align, HasDataLayout, LayoutOf, TyLayout
};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::ty::query::TyCtxtAt;
use rustc_data_structures::fx::{FxHashSet, FxHasher};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::interpret::{
    GlobalId, Scalar, FrameInfo,
    EvalResult, EvalErrorKind,
    ScalarMaybeUndef,
    truncate, sign_extend,
};

use syntax::source_map::{self, Span};
use syntax::ast::Mutability;

use super::{
    Value, Operand, MemPlace, MPlaceTy, Place, PlaceExtra,
    Memory, Machine
};

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
    /// `[return_ptr, arguments..., variables..., temporaries...]`.
    /// The locals are stored as `Option<Value>`s.
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

impl<'a, 'mir, 'tcx, M> layout::HasTyCtxt<'tcx> for &'a EvalContext<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>
{
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

    #[inline]
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
        let steps = mem::replace(&mut self.steps_since_detector_enabled,
                                 -STEPS_UNTIL_DETECTOR_ENABLED);
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

    pub(super) fn resolve(
        &self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>
    ) -> EvalResult<'tcx, ty::Instance<'tcx>> {
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
        if did.is_local()
            && self.tcx.has_typeck_tables(did)
            && self.tcx.typeck_tables_of(did).tainted_by_errors
        {
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

    /// Return the actual dynamic size and alignment of the place at the given type.
    /// Note that the value does not matter if the type is sized. For unsized types,
    /// the value has to be a fat pointer, and we only care about the "extra" data in it.
    pub fn size_and_align_of_mplace(
        &self,
        mplace: MPlaceTy<'tcx>,
    ) -> EvalResult<'tcx, (Size, Align)> {
        if let PlaceExtra::None = mplace.extra {
            assert!(!mplace.layout.is_unsized());
            Ok(mplace.layout.size_and_align())
        } else {
            let layout = mplace.layout;
            assert!(layout.is_unsized());
            match layout.ty.sty {
                ty::Adt(..) | ty::Tuple(..) => {
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
                    let field = self.mplace_field(mplace, layout.fields.count() as u64 - 1)?;
                    let (unsized_size, unsized_align) = self.size_and_align_of_mplace(field)?;

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
                ty::Dynamic(..) => {
                    let vtable = match mplace.extra {
                        PlaceExtra::Vtable(vtable) => vtable,
                        _ => bug!("Expected vtable"),
                    };
                    // the second entry in the vtable is the dynamic size of the object.
                    self.read_size_and_align_from_vtable(vtable)
                }

                ty::Slice(_) | ty::Str => {
                    let len = match mplace.extra {
                        PlaceExtra::Length(len) => len,
                        _ => bug!("Expected length"),
                    };
                    let (elem_size, align) = layout.field(self, 0)?.size_and_align();
                    Ok((elem_size * len, align))
                }

                _ => bug!("size_of_val::<{:?}> not supported", layout.ty),
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
            let mut locals = IndexVec::from_elem(dummy, &mir.local_decls);
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
                                    locals[local] = LocalValue::Dead;
                                }
                                _ => {}
                            }
                        }
                    }
                },
            }
            // Finally, properly initialize all those that still have the dummy value
            for (local, decl) in locals.iter_mut().zip(mir.local_decls.iter()) {
                match *local {
                    LocalValue::Live(_) => {
                        // This needs to be peoperly initialized.
                        let layout = self.layout_of(self.monomorphize(decl.ty, instance.substs))?;
                        *local = LocalValue::Live(self.uninit_operand(layout)?);
                    }
                    LocalValue::Dead => {
                        // Nothing to do
                    }
                }
            }
            // done
            self.frame_mut().locals = locals;
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
                    // FIXME: to_ptr()? might be too extreme here,
                    // static zsts might reach this under certain conditions
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
        self.tcx.const_eval(param_env.and(gid))
            .map_err(|err| EvalErrorKind::ReferencedConstant(err).into())
    }

    #[inline(always)]
    pub fn frame(&self) -> &Frame<'mir, 'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    #[inline(always)]
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
            let location = if self.tcx.def_key(instance.def_id()).disambiguated_data.data
                == DefPathData::ClosureExpr
            {
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

    #[inline(always)]
    pub fn sign_extend(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        assert!(ty.abi.is_signed());
        sign_extend(value, ty.size)
    }

    #[inline(always)]
    pub fn truncate(&self, value: u128, ty: TyLayout<'_>) -> u128 {
        truncate(value, ty.size)
    }
}

