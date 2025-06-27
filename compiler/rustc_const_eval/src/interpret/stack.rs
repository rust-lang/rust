//! Manages the low-level pushing and popping of stack frames and the (de)allocation of local variables.
//! For handling of argument passing and return values, see the `call` module.
use std::cell::Cell;
use std::{fmt, mem};

use either::{Either, Left, Right};
use rustc_hir as hir;
use rustc_hir::definitions::DefPathData;
use rustc_index::IndexVec;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_mir_dataflow::impls::always_storage_live_locals;
use rustc_span::Span;
use tracing::{info_span, instrument, trace};

use super::{
    AllocId, CtfeProvenance, Immediate, InterpCx, InterpResult, Machine, MemPlace, MemPlaceMeta,
    MemoryKind, Operand, PlaceTy, Pointer, Provenance, ReturnAction, Scalar, from_known_layout,
    interp_ok, throw_ub, throw_unsup,
};
use crate::errors;

// The Phantomdata exists to prevent this type from being `Send`. If it were sent across a thread
// boundary and dropped in the other thread, it would exit the span in the other thread.
struct SpanGuard(tracing::Span, std::marker::PhantomData<*const u8>);

impl SpanGuard {
    /// By default a `SpanGuard` does nothing.
    fn new() -> Self {
        Self(tracing::Span::none(), std::marker::PhantomData)
    }

    /// If a span is entered, we exit the previous span (if any, normally none) and enter the
    /// new span. This is mainly so we don't have to use `Option` for the `tracing_span` field of
    /// `Frame` by creating a dummy span to being with and then entering it once the frame has
    /// been pushed.
    fn enter(&mut self, span: tracing::Span) {
        // This executes the destructor on the previous instance of `SpanGuard`, ensuring that
        // we never enter or exit more spans than vice versa. Unless you `mem::leak`, then we
        // can't protect the tracing stack, but that'll just lead to weird logging, no actual
        // problems.
        *self = Self(span, std::marker::PhantomData);
        self.0.with_subscriber(|(id, dispatch)| {
            dispatch.enter(id);
        });
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.0.with_subscriber(|(id, dispatch)| {
            dispatch.exit(id);
        });
    }
}

/// A stack frame.
pub struct Frame<'tcx, Prov: Provenance = CtfeProvenance, Extra = ()> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub(super) body: &'tcx mir::Body<'tcx>,

    /// The def_id and args of the current function.
    pub(super) instance: ty::Instance<'tcx>,

    /// Extra data for the machine.
    pub extra: Extra,

    ////////////////////////////////////////////////////////////////////////////////
    // Return place and locals
    ////////////////////////////////////////////////////////////////////////////////
    /// Work to perform when returning from this function.
    return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to,
    /// and its layout in the caller. This place is to be interpreted relative to the
    /// *caller's* stack frame. We use a `PlaceTy` instead of an `MPlaceTy` since this
    /// avoids having to move *all* return places into Miri's memory.
    pub return_place: PlaceTy<'tcx, Prov>,

    /// The list of locals for this stack frame, stored in order as
    /// `[return_ptr, arguments..., variables..., temporaries...]`.
    /// The locals are stored as `Option<Value>`s.
    /// `None` represents a local that is currently dead, while a live local
    /// can either directly contain `Scalar` or refer to some part of an `Allocation`.
    ///
    /// Do *not* access this directly; always go through the machine hook!
    pub locals: IndexVec<mir::Local, LocalState<'tcx, Prov>>,

    /// The span of the `tracing` crate is stored here.
    /// When the guard is dropped, the span is exited. This gives us
    /// a full stack trace on all tracing statements.
    tracing_span: SpanGuard,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////
    /// If this is `Right`, we are not currently executing any particular statement in
    /// this frame (can happen e.g. during frame initialization, and during unwinding on
    /// frames without cleanup code).
    ///
    /// Needs to be public because ConstProp does unspeakable things to it.
    pub(super) loc: Either<mir::Location, Span>,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)] // Miri debug-prints these
pub enum StackPopCleanup {
    /// Jump to the next block in the caller, or cause UB if None (that's a function
    /// that may never return). Also store layout of return place so
    /// we can validate it at that layout.
    /// `ret` stores the block we jump to on a normal return, while `unwind`
    /// stores the block used for cleanup during unwinding.
    Goto { ret: Option<mir::BasicBlock>, unwind: mir::UnwindAction },
    /// The root frame of the stack: nowhere else to jump to.
    /// `cleanup` says whether locals are deallocated. Static computation
    /// wants them leaked to intern what they need (and just throw away
    /// the entire `ecx` when it is done).
    Root { cleanup: bool },
}

/// Return type of [`InterpCx::pop_stack_frame_raw`].
pub struct StackPopInfo<'tcx, Prov: Provenance> {
    /// Additional information about the action to be performed when returning from the popped
    /// stack frame.
    pub return_action: ReturnAction,

    /// [`return_to_block`](Frame::return_to_block) of the popped stack frame.
    pub return_to_block: StackPopCleanup,

    /// [`return_place`](Frame::return_place) of the popped stack frame.
    pub return_place: PlaceTy<'tcx, Prov>,
}

/// State of a local variable including a memoized layout
#[derive(Clone)]
pub struct LocalState<'tcx, Prov: Provenance = CtfeProvenance> {
    value: LocalValue<Prov>,
    /// Don't modify if `Some`, this is only used to prevent computing the layout twice.
    /// Avoids computing the layout of locals that are never actually initialized.
    layout: Cell<Option<TyAndLayout<'tcx>>>,
}

impl<Prov: Provenance> std::fmt::Debug for LocalState<'_, Prov> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalState")
            .field("value", &self.value)
            .field("ty", &self.layout.get().map(|l| l.ty))
            .finish()
    }
}

/// Current value of a local variable
///
/// This does not store the type of the local; the type is given by `body.local_decls` and can never
/// change, so by not storing here we avoid having to maintain that as an invariant.
#[derive(Copy, Clone, Debug)] // Miri debug-prints these
pub(super) enum LocalValue<Prov: Provenance = CtfeProvenance> {
    /// This local is not currently alive, and cannot be used at all.
    Dead,
    /// A normal, live local.
    /// Mostly for convenience, we re-use the `Operand` type here.
    /// This is an optimization over just always having a pointer here;
    /// we can thus avoid doing an allocation when the local just stores
    /// immediate values *and* never has its address taken.
    Live(Operand<Prov>),
}

impl<'tcx, Prov: Provenance> LocalState<'tcx, Prov> {
    pub fn make_live_uninit(&mut self) {
        self.value = LocalValue::Live(Operand::Immediate(Immediate::Uninit));
    }

    /// This is a hack because Miri needs a way to visit all the provenance in a `LocalState`
    /// without having a layout or `TyCtxt` available, and we want to keep the `Operand` type
    /// private.
    pub fn as_mplace_or_imm(
        &self,
    ) -> Option<Either<(Pointer<Option<Prov>>, MemPlaceMeta<Prov>), Immediate<Prov>>> {
        match self.value {
            LocalValue::Dead => None,
            LocalValue::Live(Operand::Indirect(mplace)) => Some(Left((mplace.ptr, mplace.meta))),
            LocalValue::Live(Operand::Immediate(imm)) => Some(Right(imm)),
        }
    }

    /// Read the local's value or error if the local is not yet live or not live anymore.
    #[inline(always)]
    pub(super) fn access(&self) -> InterpResult<'tcx, &Operand<Prov>> {
        match &self.value {
            LocalValue::Dead => throw_ub!(DeadLocal), // could even be "invalid program"?
            LocalValue::Live(val) => interp_ok(val),
        }
    }

    /// Overwrite the local. If the local can be overwritten in place, return a reference
    /// to do so; otherwise return the `MemPlace` to consult instead.
    #[inline(always)]
    pub(super) fn access_mut(&mut self) -> InterpResult<'tcx, &mut Operand<Prov>> {
        match &mut self.value {
            LocalValue::Dead => throw_ub!(DeadLocal), // could even be "invalid program"?
            LocalValue::Live(val) => interp_ok(val),
        }
    }
}

/// What we store about a frame in an interpreter backtrace.
#[derive(Clone, Debug)]
pub struct FrameInfo<'tcx> {
    pub instance: ty::Instance<'tcx>,
    pub span: Span,
}

// FIXME: only used by miri, should be removed once translatable.
impl<'tcx> fmt::Display for FrameInfo<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            if tcx.def_key(self.instance.def_id()).disambiguated_data.data == DefPathData::Closure {
                write!(f, "inside closure")
            } else {
                // Note: this triggers a `must_produce_diag` state, which means that if we ever
                // get here we must emit a diagnostic. We should never display a `FrameInfo` unless
                // we actually want to emit a warning or error to the user.
                write!(f, "inside `{}`", self.instance)
            }
        })
    }
}

impl<'tcx> FrameInfo<'tcx> {
    pub fn as_note(&self, tcx: TyCtxt<'tcx>) -> errors::FrameNote {
        let span = self.span;
        if tcx.def_key(self.instance.def_id()).disambiguated_data.data == DefPathData::Closure {
            errors::FrameNote {
                where_: "closure",
                span,
                instance: String::new(),
                times: 0,
                has_label: false,
            }
        } else {
            let instance = format!("{}", self.instance);
            // Note: this triggers a `must_produce_diag` state, which means that if we ever get
            // here we must emit a diagnostic. We should never display a `FrameInfo` unless we
            // actually want to emit a warning or error to the user.
            errors::FrameNote { where_: "instance", span, instance, times: 0, has_label: false }
        }
    }
}

impl<'tcx, Prov: Provenance> Frame<'tcx, Prov> {
    pub fn with_extra<Extra>(self, extra: Extra) -> Frame<'tcx, Prov, Extra> {
        Frame {
            body: self.body,
            instance: self.instance,
            return_to_block: self.return_to_block,
            return_place: self.return_place,
            locals: self.locals,
            loc: self.loc,
            extra,
            tracing_span: self.tracing_span,
        }
    }
}

impl<'tcx, Prov: Provenance, Extra> Frame<'tcx, Prov, Extra> {
    /// Get the current location within the Frame.
    ///
    /// If this is `Right`, we are not currently executing any particular statement in
    /// this frame (can happen e.g. during frame initialization, and during unwinding on
    /// frames without cleanup code).
    ///
    /// Used by [priroda](https://github.com/oli-obk/priroda).
    pub fn current_loc(&self) -> Either<mir::Location, Span> {
        self.loc
    }

    pub fn body(&self) -> &'tcx mir::Body<'tcx> {
        self.body
    }

    pub fn instance(&self) -> ty::Instance<'tcx> {
        self.instance
    }

    /// Return the `SourceInfo` of the current instruction.
    pub fn current_source_info(&self) -> Option<&mir::SourceInfo> {
        self.loc.left().map(|loc| self.body.source_info(loc))
    }

    pub fn current_span(&self) -> Span {
        match self.loc {
            Left(loc) => self.body.source_info(loc).span,
            Right(span) => span,
        }
    }

    pub fn lint_root(&self, tcx: TyCtxt<'tcx>) -> Option<hir::HirId> {
        // We first try to get a HirId via the current source scope,
        // and fall back to `body.source`.
        self.current_source_info()
            .and_then(|source_info| match &self.body.source_scopes[source_info.scope].local_data {
                mir::ClearCrossCrate::Set(data) => Some(data.lint_root),
                mir::ClearCrossCrate::Clear => None,
            })
            .or_else(|| {
                let def_id = self.body.source.def_id().as_local();
                def_id.map(|def_id| tcx.local_def_id_to_hir_id(def_id))
            })
    }

    /// Returns the address of the buffer where the locals are stored. This is used by `Place` as a
    /// sanity check to detect bugs where we mix up which stack frame a place refers to.
    #[inline(always)]
    pub(super) fn locals_addr(&self) -> usize {
        self.locals.raw.as_ptr().addr()
    }

    #[must_use]
    pub fn generate_stacktrace_from_stack(stack: &[Self]) -> Vec<FrameInfo<'tcx>> {
        let mut frames = Vec::new();
        // This deliberately does *not* honor `requires_caller_location` since it is used for much
        // more than just panics.
        for frame in stack.iter().rev() {
            let span = match frame.loc {
                Left(loc) => {
                    // If the stacktrace passes through MIR-inlined source scopes, add them.
                    let mir::SourceInfo { mut span, scope } = *frame.body.source_info(loc);
                    let mut scope_data = &frame.body.source_scopes[scope];
                    while let Some((instance, call_span)) = scope_data.inlined {
                        frames.push(FrameInfo { span, instance });
                        span = call_span;
                        scope_data = &frame.body.source_scopes[scope_data.parent_scope.unwrap()];
                    }
                    span
                }
                Right(span) => span,
            };
            frames.push(FrameInfo { span, instance: frame.instance });
        }
        trace!("generate stacktrace: {:#?}", frames);
        frames
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Very low-level helper that pushes a stack frame without initializing
    /// the arguments or local variables.
    ///
    /// The high-level version of this is `init_stack_frame`.
    #[instrument(skip(self, body, return_place, return_to_block), level = "debug")]
    pub(crate) fn push_stack_frame_raw(
        &mut self,
        instance: ty::Instance<'tcx>,
        body: &'tcx mir::Body<'tcx>,
        return_place: &PlaceTy<'tcx, M::Provenance>,
        return_to_block: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        trace!("body: {:#?}", body);

        // We can push a `Root` frame if and only if the stack is empty.
        debug_assert_eq!(
            self.stack().is_empty(),
            matches!(return_to_block, StackPopCleanup::Root { .. })
        );

        // First push a stack frame so we have access to `instantiate_from_current_frame` and other
        // `self.frame()`-based functions.
        let dead_local = LocalState { value: LocalValue::Dead, layout: Cell::new(None) };
        let locals = IndexVec::from_elem(dead_local, &body.local_decls);
        let pre_frame = Frame {
            body,
            loc: Right(body.span), // Span used for errors caused during preamble.
            return_to_block,
            return_place: return_place.clone(),
            locals,
            instance,
            tracing_span: SpanGuard::new(),
            extra: (),
        };
        let frame = M::init_frame(self, pre_frame)?;
        self.stack_mut().push(frame);

        // Make sure all the constants required by this frame evaluate successfully (post-monomorphization check).
        for &const_ in body.required_consts() {
            let c =
                self.instantiate_from_current_frame_and_normalize_erasing_regions(const_.const_)?;
            c.eval(*self.tcx, self.typing_env, const_.span).map_err(|err| {
                err.emit_note(*self.tcx);
                err
            })?;
        }

        // Finish things up.
        M::after_stack_push(self)?;
        self.frame_mut().loc = Left(mir::Location::START);
        let span = info_span!("frame", "{}", instance);
        self.frame_mut().tracing_span.enter(span);

        interp_ok(())
    }

    /// Low-level helper that pops a stack frame from the stack and returns some information about
    /// it.
    ///
    /// This also deallocates locals, if necessary.
    /// `copy_ret_val` gets called after the frame has been taken from the stack but before the locals have been deallocated.
    ///
    /// [`M::before_stack_pop`] and [`M::after_stack_pop`] are called by this function
    /// automatically.
    ///
    /// The high-level version of this is `return_from_current_stack_frame`.
    ///
    /// [`M::before_stack_pop`]: Machine::before_stack_pop
    /// [`M::after_stack_pop`]: Machine::after_stack_pop
    pub(super) fn pop_stack_frame_raw(
        &mut self,
        unwinding: bool,
        copy_ret_val: impl FnOnce(&mut Self, &PlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, StackPopInfo<'tcx, M::Provenance>> {
        M::before_stack_pop(self)?;
        let frame =
            self.stack_mut().pop().expect("tried to pop a stack frame, but there were none");

        // Copy return value (unless we are unwinding).
        if !unwinding {
            copy_ret_val(self, &frame.return_place)?;
        }

        let return_to_block = frame.return_to_block;
        let return_place = frame.return_place.clone();

        // Cleanup: deallocate locals.
        // Usually we want to clean up (deallocate locals), but in a few rare cases we don't.
        // We do this while the frame is still on the stack, so errors point to the callee.
        let cleanup = match return_to_block {
            StackPopCleanup::Goto { .. } => true,
            StackPopCleanup::Root { cleanup, .. } => cleanup,
        };

        let return_action = if cleanup {
            // We need to take the locals out, since we need to mutate while iterating.
            for local in &frame.locals {
                self.deallocate_local(local.value)?;
            }

            // Call the machine hook, which determines the next steps.
            let return_action = M::after_stack_pop(self, frame, unwinding)?;
            assert_ne!(return_action, ReturnAction::NoCleanup);
            return_action
        } else {
            // We also skip the machine hook when there's no cleanup. This not a real "pop" anyway.
            ReturnAction::NoCleanup
        };

        interp_ok(StackPopInfo { return_action, return_to_block, return_place })
    }

    /// In the current stack frame, mark all locals as live that are not arguments and don't have
    /// `Storage*` annotations (this includes the return place).
    pub(crate) fn storage_live_for_always_live_locals(&mut self) -> InterpResult<'tcx> {
        self.storage_live(mir::RETURN_PLACE)?;

        let body = self.body();
        let always_live = always_storage_live_locals(body);
        for local in body.vars_and_temps_iter() {
            if always_live.contains(local) {
                self.storage_live(local)?;
            }
        }
        interp_ok(())
    }

    pub fn storage_live_dyn(
        &mut self,
        local: mir::Local,
        meta: MemPlaceMeta<M::Provenance>,
    ) -> InterpResult<'tcx> {
        trace!("{:?} is now live", local);

        // We avoid `ty.is_trivially_sized` since that does something expensive for ADTs.
        fn is_very_trivially_sized(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
                | ty::Uint(_)
                | ty::Int(_)
                | ty::Bool
                | ty::Float(_)
                | ty::FnDef(..)
                | ty::FnPtr(..)
                | ty::RawPtr(..)
                | ty::Char
                | ty::Ref(..)
                | ty::Coroutine(..)
                | ty::CoroutineWitness(..)
                | ty::Array(..)
                | ty::Closure(..)
                | ty::CoroutineClosure(..)
                | ty::Never
                | ty::Error(_) => true,

                ty::Str | ty::Slice(_) | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => false,

                ty::Tuple(tys) => tys.last().is_none_or(|ty| is_very_trivially_sized(*ty)),

                ty::Pat(ty, ..) => is_very_trivially_sized(*ty),

                // We don't want to do any queries, so there is not much we can do with ADTs.
                ty::Adt(..) => false,

                ty::UnsafeBinder(ty) => is_very_trivially_sized(ty.skip_binder()),

                ty::Alias(..) | ty::Param(_) | ty::Placeholder(..) => false,

                ty::Infer(ty::TyVar(_)) => false,

                ty::Bound(..)
                | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                    bug!("`is_very_trivially_sized` applied to unexpected type: {}", ty)
                }
            }
        }

        // This is a hot function, we avoid computing the layout when possible.
        // `unsized_` will be `None` for sized types and `Some(layout)` for unsized types.
        let unsized_ = if is_very_trivially_sized(self.body().local_decls[local].ty) {
            None
        } else {
            // We need the layout.
            let layout = self.layout_of_local(self.frame(), local, None)?;
            if layout.is_sized() { None } else { Some(layout) }
        };

        let local_val = LocalValue::Live(if let Some(layout) = unsized_ {
            if !meta.has_meta() {
                throw_unsup!(UnsizedLocal);
            }
            // Need to allocate some memory, since `Immediate::Uninit` cannot be unsized.
            let dest_place = self.allocate_dyn(layout, MemoryKind::Stack, meta)?;
            Operand::Indirect(*dest_place.mplace())
        } else {
            // Just make this an efficient immediate.
            assert!(!meta.has_meta()); // we're dropping the metadata
            // Make sure the machine knows this "write" is happening. (This is important so that
            // races involving local variable allocation can be detected by Miri.)
            M::after_local_write(self, local, /*storage_live*/ true)?;
            // Note that not calling `layout_of` here does have one real consequence:
            // if the type is too big, we'll only notice this when the local is actually initialized,
            // which is a bit too late -- we should ideally notice this already here, when the memory
            // is conceptually allocated. But given how rare that error is and that this is a hot function,
            // we accept this downside for now.
            Operand::Immediate(Immediate::Uninit)
        });

        // If the local is already live, deallocate its old memory.
        let old = mem::replace(&mut self.frame_mut().locals[local].value, local_val);
        self.deallocate_local(old)?;
        interp_ok(())
    }

    /// Mark a storage as live, killing the previous content.
    #[inline(always)]
    pub fn storage_live(&mut self, local: mir::Local) -> InterpResult<'tcx> {
        self.storage_live_dyn(local, MemPlaceMeta::None)
    }

    pub fn storage_dead(&mut self, local: mir::Local) -> InterpResult<'tcx> {
        assert!(local != mir::RETURN_PLACE, "Cannot make return place dead");
        trace!("{:?} is now dead", local);

        // If the local is already dead, this is a NOP.
        let old = mem::replace(&mut self.frame_mut().locals[local].value, LocalValue::Dead);
        self.deallocate_local(old)?;
        interp_ok(())
    }

    fn deallocate_local(&mut self, local: LocalValue<M::Provenance>) -> InterpResult<'tcx> {
        if let LocalValue::Live(Operand::Indirect(MemPlace { ptr, .. })) = local {
            // All locals have a backing allocation, even if the allocation is empty
            // due to the local having ZST type. Hence we can `unwrap`.
            trace!(
                "deallocating local {:?}: {:?}",
                local,
                // Locals always have a `alloc_id` (they are never the result of a int2ptr).
                self.dump_alloc(ptr.provenance.unwrap().get_alloc_id().unwrap())
            );
            self.deallocate_ptr(ptr, None, MemoryKind::Stack)?;
        };
        interp_ok(())
    }

    /// This is public because it is used by [Aquascope](https://github.com/cognitive-engineering-lab/aquascope/)
    /// to analyze all the locals in a stack frame.
    #[inline(always)]
    pub fn layout_of_local(
        &self,
        frame: &Frame<'tcx, M::Provenance, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let state = &frame.locals[local];
        if let Some(layout) = state.layout.get() {
            return interp_ok(layout);
        }

        let layout = from_known_layout(self.tcx, self.typing_env, layout, || {
            let local_ty = frame.body.local_decls[local].ty;
            let local_ty =
                self.instantiate_from_frame_and_normalize_erasing_regions(frame, local_ty)?;
            self.layout_of(local_ty).into()
        })?;

        // Layouts of locals are requested a lot, so we cache them.
        state.layout.set(Some(layout));
        interp_ok(layout)
    }
}

impl<'tcx, Prov: Provenance> LocalState<'tcx, Prov> {
    pub(super) fn print(
        &self,
        allocs: &mut Vec<Option<AllocId>>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self.value {
            LocalValue::Dead => write!(fmt, " is dead")?,
            LocalValue::Live(Operand::Immediate(Immediate::Uninit)) => {
                write!(fmt, " is uninitialized")?
            }
            LocalValue::Live(Operand::Indirect(mplace)) => {
                write!(
                    fmt,
                    " by {} ref {:?}:",
                    match mplace.meta {
                        MemPlaceMeta::Meta(meta) => format!(" meta({meta:?})"),
                        MemPlaceMeta::None => String::new(),
                    },
                    mplace.ptr,
                )?;
                allocs.extend(mplace.ptr.provenance.map(Provenance::get_alloc_id));
            }
            LocalValue::Live(Operand::Immediate(Immediate::Scalar(val))) => {
                write!(fmt, " {val:?}")?;
                if let Scalar::Ptr(ptr, _size) = val {
                    allocs.push(ptr.provenance.get_alloc_id());
                }
            }
            LocalValue::Live(Operand::Immediate(Immediate::ScalarPair(val1, val2))) => {
                write!(fmt, " ({val1:?}, {val2:?})")?;
                if let Scalar::Ptr(ptr, _size) = val1 {
                    allocs.push(ptr.provenance.get_alloc_id());
                }
                if let Scalar::Ptr(ptr, _size) = val2 {
                    allocs.push(ptr.provenance.get_alloc_id());
                }
            }
        }

        Ok(())
    }
}
