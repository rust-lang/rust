use std::cell::Cell;
use std::{fmt, mem};

use either::{Either, Left, Right};

use hir::CRATE_HIR_ID;
use rustc_errors::DiagCtxt;
use rustc_hir::{self as hir, def_id::DefId, definitions::DefPathData};
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{
    CtfeProvenance, ErrorHandled, InvalidMetaKind, ReportedErrorInfo,
};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::{
    self, FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOf, LayoutOfHelpers,
    TyAndLayout,
};
use rustc_middle::ty::{self, GenericArgsRef, ParamEnv, Ty, TyCtxt, TypeFoldable, Variance};
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_session::Limit;
use rustc_span::Span;
use rustc_target::abi::{call::FnAbi, Align, HasDataLayout, Size, TargetDataLayout};

use super::{
    GlobalId, Immediate, InterpErrorInfo, InterpResult, MPlaceTy, Machine, MemPlace, MemPlaceMeta,
    Memory, MemoryKind, OpTy, Operand, Place, PlaceTy, Pointer, PointerArithmetic, Projectable,
    Provenance, Scalar, StackPopJump,
};
use crate::errors;
use crate::util;
use crate::{fluent_generated as fluent, ReportErrorExt};

pub struct InterpCx<'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// Stores the `Machine` instance.
    ///
    /// Note: the stack is provided by the machine.
    pub machine: M,

    /// The results of the type checker, from rustc.
    /// The span in this is the "root" of the evaluation, i.e., the const
    /// we are evaluating (if this is CTFE).
    pub tcx: TyCtxtAt<'tcx>,

    /// Bounds in scope for polymorphic evaluations.
    pub(crate) param_env: ty::ParamEnv<'tcx>,

    /// The virtual memory system.
    pub memory: Memory<'mir, 'tcx, M>,

    /// The recursion limit (cached from `tcx.recursion_limit(())`)
    pub recursion_limit: Limit,
}

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
pub struct Frame<'mir, 'tcx, Prov: Provenance = CtfeProvenance, Extra = ()> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////
    /// The MIR for the function called on this frame.
    pub body: &'mir mir::Body<'tcx>,

    /// The def_id and args of the current function.
    pub instance: ty::Instance<'tcx>,

    /// Extra data for the machine.
    pub extra: Extra,

    ////////////////////////////////////////////////////////////////////////////////
    // Return place and locals
    ////////////////////////////////////////////////////////////////////////////////
    /// Work to perform when returning from this function.
    pub return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to,
    /// and its layout in the caller.
    pub return_place: MPlaceTy<'tcx, Prov>,

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
    pub loc: Either<mir::Location, Span>,
}

/// What we store about a frame in an interpreter backtrace.
#[derive(Clone, Debug)]
pub struct FrameInfo<'tcx> {
    pub instance: ty::Instance<'tcx>,
    pub span: Span,
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
            LocalValue::Live(val) => Ok(val),
        }
    }

    /// Overwrite the local. If the local can be overwritten in place, return a reference
    /// to do so; otherwise return the `MemPlace` to consult instead.
    ///
    /// Note: Before calling this, call the `before_access_local_mut` machine hook! You may be
    /// invalidating machine invariants otherwise!
    #[inline(always)]
    pub(super) fn access_mut(&mut self) -> InterpResult<'tcx, &mut Operand<Prov>> {
        match &mut self.value {
            LocalValue::Dead => throw_ub!(DeadLocal), // could even be "invalid program"?
            LocalValue::Live(val) => Ok(val),
        }
    }
}

impl<'mir, 'tcx, Prov: Provenance> Frame<'mir, 'tcx, Prov> {
    pub fn with_extra<Extra>(self, extra: Extra) -> Frame<'mir, 'tcx, Prov, Extra> {
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

impl<'mir, 'tcx, Prov: Provenance, Extra> Frame<'mir, 'tcx, Prov, Extra> {
    /// Get the current location within the Frame.
    ///
    /// If this is `Left`, we are not currently executing any particular statement in
    /// this frame (can happen e.g. during frame initialization, and during unwinding on
    /// frames without cleanup code).
    ///
    /// Used by priroda.
    pub fn current_loc(&self) -> Either<mir::Location, Span> {
        self.loc
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

    pub fn lint_root(&self) -> Option<hir::HirId> {
        self.current_source_info().and_then(|source_info| {
            match &self.body.source_scopes[source_info.scope].local_data {
                mir::ClearCrossCrate::Set(data) => Some(data.lint_root),
                mir::ClearCrossCrate::Clear => None,
            }
        })
    }
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
            errors::FrameNote { where_: "closure", span, instance: String::new(), times: 0 }
        } else {
            let instance = format!("{}", self.instance);
            // Note: this triggers a `must_produce_diag` state, which means that if we ever get
            // here we must emit a diagnostic. We should never display a `FrameInfo` unless we
            // actually want to emit a warning or error to the user.
            errors::FrameNote { where_: "instance", span, instance, times: 0 }
        }
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for InterpCx<'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
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

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> LayoutOfHelpers<'tcx> for InterpCx<'mir, 'tcx, M> {
    type LayoutOfResult = InterpResult<'tcx, TyAndLayout<'tcx>>;

    #[inline]
    fn layout_tcx_at_span(&self) -> Span {
        // Using the cheap root span for performance.
        self.tcx.span
    }

    #[inline]
    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        _: Span,
        _: Ty<'tcx>,
    ) -> InterpErrorInfo<'tcx> {
        err_inval!(Layout(err)).into()
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> FnAbiOfHelpers<'tcx> for InterpCx<'mir, 'tcx, M> {
    type FnAbiOfResult = InterpResult<'tcx, &'tcx FnAbi<'tcx, Ty<'tcx>>>;

    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        _span: Span,
        _fn_abi_request: FnAbiRequest<'tcx>,
    ) -> InterpErrorInfo<'tcx> {
        match err {
            FnAbiError::Layout(err) => err_inval!(Layout(err)).into(),
            FnAbiError::AdjustForForeignAbi(err) => {
                err_inval!(FnAbiAdjustForForeignAbi(err)).into()
            }
        }
    }
}

/// Test if it is valid for a MIR assignment to assign `src`-typed place to `dest`-typed value.
/// This test should be symmetric, as it is primarily about layout compatibility.
pub(super) fn mir_assign_valid_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    src: TyAndLayout<'tcx>,
    dest: TyAndLayout<'tcx>,
) -> bool {
    // Type-changing assignments can happen when subtyping is used. While
    // all normal lifetimes are erased, higher-ranked types with their
    // late-bound lifetimes are still around and can lead to type
    // differences.
    if util::relate_types(tcx, param_env, Variance::Covariant, src.ty, dest.ty) {
        // Make sure the layout is equal, too -- just to be safe. Miri really
        // needs layout equality. For performance reason we skip this check when
        // the types are equal. Equal types *can* have different layouts when
        // enum downcast is involved (as enum variants carry the type of the
        // enum), but those should never occur in assignments.
        if cfg!(debug_assertions) || src.ty != dest.ty {
            assert_eq!(src.layout, dest.layout);
        }
        true
    } else {
        false
    }
}

/// Use the already known layout if given (but sanity check in debug mode),
/// or compute the layout.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn from_known_layout<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    param_env: ParamEnv<'tcx>,
    known_layout: Option<TyAndLayout<'tcx>>,
    compute: impl FnOnce() -> InterpResult<'tcx, TyAndLayout<'tcx>>,
) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
    match known_layout {
        None => compute(),
        Some(known_layout) => {
            if cfg!(debug_assertions) {
                let check_layout = compute()?;
                if !mir_assign_valid_types(tcx.tcx, param_env, check_layout, known_layout) {
                    span_bug!(
                        tcx.span,
                        "expected type differs from actual type.\nexpected: {}\nactual: {}",
                        known_layout.ty,
                        check_layout.ty,
                    );
                }
            }
            Ok(known_layout)
        }
    }
}

/// Turn the given error into a human-readable string. Expects the string to be printed, so if
/// `RUSTC_CTFE_BACKTRACE` is set this will show a backtrace of the rustc internals that
/// triggered the error.
///
/// This is NOT the preferred way to render an error; use `report` from `const_eval` instead.
/// However, this is useful when error messages appear in ICEs.
pub fn format_interp_error<'tcx>(dcx: &DiagCtxt, e: InterpErrorInfo<'tcx>) -> String {
    let (e, backtrace) = e.into_parts();
    backtrace.print_backtrace();
    // FIXME(fee1-dead), HACK: we want to use the error as title therefore we can just extract the
    // label and arguments from the InterpError.
    #[allow(rustc::untranslatable_diagnostic)]
    let mut diag = dcx.struct_allow("");
    let msg = e.diagnostic_message();
    e.add_args(&mut diag);
    let s = dcx.eagerly_translate_to_string(msg, diag.args.iter());
    diag.cancel();
    s
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        root_span: Span,
        param_env: ty::ParamEnv<'tcx>,
        machine: M,
    ) -> Self {
        InterpCx {
            machine,
            tcx: tcx.at(root_span),
            param_env,
            memory: Memory::new(),
            recursion_limit: tcx.recursion_limit(),
        }
    }

    #[inline(always)]
    pub fn cur_span(&self) -> Span {
        // This deliberately does *not* honor `requires_caller_location` since it is used for much
        // more than just panics.
        self.stack().last().map_or(self.tcx.span, |f| f.current_span())
    }

    #[inline(always)]
    /// Find the first stack frame that is within the current crate, if any, otherwise return the crate's HirId
    pub fn best_lint_scope(&self) -> hir::HirId {
        self.stack()
            .iter()
            .find_map(|frame| frame.body.source.def_id().as_local())
            .map_or(CRATE_HIR_ID, |def_id| self.tcx.local_def_id_to_hir_id(def_id))
    }

    #[inline(always)]
    pub(crate) fn stack(&self) -> &[Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>] {
        M::stack(self)
    }

    #[inline(always)]
    pub(crate) fn stack_mut(
        &mut self,
    ) -> &mut Vec<Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>> {
        M::stack_mut(self)
    }

    #[inline(always)]
    pub fn frame_idx(&self) -> usize {
        let stack = self.stack();
        assert!(!stack.is_empty());
        stack.len() - 1
    }

    #[inline(always)]
    pub fn frame(&self) -> &Frame<'mir, 'tcx, M::Provenance, M::FrameExtra> {
        self.stack().last().expect("no call frames exist")
    }

    #[inline(always)]
    pub fn frame_mut(&mut self) -> &mut Frame<'mir, 'tcx, M::Provenance, M::FrameExtra> {
        self.stack_mut().last_mut().expect("no call frames exist")
    }

    #[inline(always)]
    pub fn body(&self) -> &'mir mir::Body<'tcx> {
        self.frame().body
    }

    #[inline(always)]
    pub fn sign_extend(&self, value: u128, ty: TyAndLayout<'_>) -> u128 {
        assert!(ty.abi.is_signed());
        ty.size.sign_extend(value)
    }

    #[inline(always)]
    pub fn truncate(&self, value: u128, ty: TyAndLayout<'_>) -> u128 {
        ty.size.truncate(value)
    }

    #[inline]
    pub fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        ty.is_freeze(*self.tcx, self.param_env)
    }

    pub fn load_mir(
        &self,
        instance: ty::InstanceDef<'tcx>,
        promoted: Option<mir::Promoted>,
    ) -> InterpResult<'tcx, &'tcx mir::Body<'tcx>> {
        trace!("load mir(instance={:?}, promoted={:?})", instance, promoted);
        let body = if let Some(promoted) = promoted {
            let def = instance.def_id();
            &self.tcx.promoted_mir(def)[promoted]
        } else {
            M::load_mir(self, instance)?
        };
        // do not continue if typeck errors occurred (can only occur in local crate)
        if let Some(err) = body.tainted_by_errors {
            throw_inval!(AlreadyReported(ReportedErrorInfo::tainted_by_errors(err)));
        }
        Ok(body)
    }

    /// Call this on things you got out of the MIR (so it is as generic as the current
    /// stack frame), to bring it into the proper environment for this interpreter.
    pub(super) fn instantiate_from_current_frame_and_normalize_erasing_regions<
        T: TypeFoldable<TyCtxt<'tcx>>,
    >(
        &self,
        value: T,
    ) -> Result<T, ErrorHandled> {
        self.instantiate_from_frame_and_normalize_erasing_regions(self.frame(), value)
    }

    /// Call this on things you got out of the MIR (so it is as generic as the provided
    /// stack frame), to bring it into the proper environment for this interpreter.
    pub(super) fn instantiate_from_frame_and_normalize_erasing_regions<
        T: TypeFoldable<TyCtxt<'tcx>>,
    >(
        &self,
        frame: &Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>,
        value: T,
    ) -> Result<T, ErrorHandled> {
        frame
            .instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                *self.tcx,
                self.param_env,
                ty::EarlyBinder::bind(value),
            )
            .map_err(|_| ErrorHandled::TooGeneric(self.cur_span()))
    }

    /// The `args` are assumed to already be in our interpreter "universe" (param_env).
    pub(super) fn resolve(
        &self,
        def: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> InterpResult<'tcx, ty::Instance<'tcx>> {
        trace!("resolve: {:?}, {:#?}", def, args);
        trace!("param_env: {:#?}", self.param_env);
        trace!("args: {:#?}", args);
        match ty::Instance::resolve(*self.tcx, self.param_env, def, args) {
            Ok(Some(instance)) => Ok(instance),
            Ok(None) => throw_inval!(TooGeneric),

            // FIXME(eddyb) this could be a bit more specific than `AlreadyReported`.
            Err(error_reported) => throw_inval!(AlreadyReported(error_reported.into())),
        }
    }

    /// Walks up the callstack from the intrinsic's callsite, searching for the first callsite in a
    /// frame which is not `#[track_caller]`. This is the fancy version of `cur_span`.
    pub(crate) fn find_closest_untracked_caller_location(&self) -> Span {
        for frame in self.stack().iter().rev() {
            debug!("find_closest_untracked_caller_location: checking frame {:?}", frame.instance);

            // Assert that the frame we look at is actually executing code currently
            // (`loc` is `Right` when we are unwinding and the frame does not require cleanup).
            let loc = frame.loc.left().unwrap();

            // This could be a non-`Call` terminator (such as `Drop`), or not a terminator at all
            // (such as `box`). Use the normal span by default.
            let mut source_info = *frame.body.source_info(loc);

            // If this is a `Call` terminator, use the `fn_span` instead.
            let block = &frame.body.basic_blocks[loc.block];
            if loc.statement_index == block.statements.len() {
                debug!(
                    "find_closest_untracked_caller_location: got terminator {:?} ({:?})",
                    block.terminator(),
                    block.terminator().kind,
                );
                if let mir::TerminatorKind::Call { fn_span, .. } = block.terminator().kind {
                    source_info.span = fn_span;
                }
            }

            let caller_location = if frame.instance.def.requires_caller_location(*self.tcx) {
                // We use `Err(())` as indication that we should continue up the call stack since
                // this is a `#[track_caller]` function.
                Some(Err(()))
            } else {
                None
            };
            if let Ok(span) =
                frame.body.caller_location_span(source_info, caller_location, *self.tcx, Ok)
            {
                return span;
            }
        }

        span_bug!(self.cur_span(), "no non-`#[track_caller]` frame found")
    }

    #[inline(always)]
    pub fn layout_of_local(
        &self,
        frame: &Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let state = &frame.locals[local];
        if let Some(layout) = state.layout.get() {
            return Ok(layout);
        }

        let layout = from_known_layout(self.tcx, self.param_env, layout, || {
            let local_ty = frame.body.local_decls[local].ty;
            let local_ty =
                self.instantiate_from_frame_and_normalize_erasing_regions(frame, local_ty)?;
            self.layout_of(local_ty)
        })?;

        // Layouts of locals are requested a lot, so we cache them.
        state.layout.set(Some(layout));
        Ok(layout)
    }

    /// Returns the actual dynamic size and alignment of the place at the given type.
    /// Only the "meta" (metadata) part of the place matters.
    /// This can fail to provide an answer for extern types.
    pub(super) fn size_and_align_of(
        &self,
        metadata: &MemPlaceMeta<M::Provenance>,
        layout: &TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, Option<(Size, Align)>> {
        if layout.is_sized() {
            return Ok(Some((layout.size, layout.align.abi)));
        }
        match layout.ty.kind() {
            ty::Adt(..) | ty::Tuple(..) => {
                // First get the size of all statically known fields.
                // Don't use type_of::sizing_type_of because that expects t to be sized,
                // and it also rounds up to alignment, which we want to avoid,
                // as the unsized field's alignment could be smaller.
                assert!(!layout.ty.is_simd());
                assert!(layout.fields.count() > 0);
                trace!("DST layout: {:?}", layout);

                let unsized_offset_unadjusted = layout.fields.offset(layout.fields.count() - 1);
                let sized_align = layout.align.abi;

                // Recurse to get the size of the dynamically sized field (must be
                // the last field). Can't have foreign types here, how would we
                // adjust alignment and size for them?
                let field = layout.field(self, layout.fields.count() - 1);
                let Some((unsized_size, mut unsized_align)) =
                    self.size_and_align_of(metadata, &field)?
                else {
                    // A field with an extern type. We don't know the actual dynamic size
                    // or the alignment.
                    return Ok(None);
                };

                // # First compute the dynamic alignment

                // Packed type alignment needs to be capped.
                if let ty::Adt(def, _) = layout.ty.kind() {
                    if let Some(packed) = def.repr().pack {
                        unsized_align = unsized_align.min(packed);
                    }
                }

                // Choose max of two known alignments (combined value must
                // be aligned according to more restrictive of the two).
                let full_align = sized_align.max(unsized_align);

                // # Then compute the dynamic size

                let unsized_offset_adjusted = unsized_offset_unadjusted.align_to(unsized_align);
                let full_size = (unsized_offset_adjusted + unsized_size).align_to(full_align);

                // Just for our sanitiy's sake, assert that this is equal to what codegen would compute.
                assert_eq!(
                    full_size,
                    (unsized_offset_unadjusted + unsized_size).align_to(full_align)
                );

                // Check if this brought us over the size limit.
                if full_size > self.max_size_of_val() {
                    throw_ub!(InvalidMeta(InvalidMetaKind::TooBig));
                }
                Ok(Some((full_size, full_align)))
            }
            ty::Dynamic(_, _, ty::Dyn) => {
                let vtable = metadata.unwrap_meta().to_pointer(self)?;
                // Read size and align from vtable (already checks size).
                Ok(Some(self.get_vtable_size_and_align(vtable)?))
            }

            ty::Slice(_) | ty::Str => {
                let len = metadata.unwrap_meta().to_target_usize(self)?;
                let elem = layout.field(self, 0);

                // Make sure the slice is not too big.
                let size = elem.size.bytes().saturating_mul(len); // we rely on `max_size_of_val` being smaller than `u64::MAX`.
                let size = Size::from_bytes(size);
                if size > self.max_size_of_val() {
                    throw_ub!(InvalidMeta(InvalidMetaKind::SliceTooBig));
                }
                Ok(Some((size, elem.align.abi)))
            }

            ty::Foreign(_) => Ok(None),

            _ => span_bug!(self.cur_span(), "size_and_align_of::<{}> not supported", layout.ty),
        }
    }
    #[inline]
    pub fn size_and_align_of_mplace(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Option<(Size, Align)>> {
        self.size_and_align_of(&mplace.meta(), &mplace.layout)
    }

    #[instrument(skip(self, body, return_place, return_to_block), level = "debug")]
    pub fn push_stack_frame(
        &mut self,
        instance: ty::Instance<'tcx>,
        body: &'mir mir::Body<'tcx>,
        return_place: &MPlaceTy<'tcx, M::Provenance>,
        return_to_block: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        trace!("body: {:#?}", body);
        let dead_local = LocalState { value: LocalValue::Dead, layout: Cell::new(None) };
        let locals = IndexVec::from_elem(dead_local, &body.local_decls);
        // First push a stack frame so we have access to the local args
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
        let frame = M::init_frame_extra(self, pre_frame)?;
        self.stack_mut().push(frame);

        // Make sure all the constants required by this frame evaluate successfully (post-monomorphization check).
        if M::POST_MONO_CHECKS {
            for &const_ in &body.required_consts {
                let c = self
                    .instantiate_from_current_frame_and_normalize_erasing_regions(const_.const_)?;
                c.eval(*self.tcx, self.param_env, Some(const_.span)).map_err(|err| {
                    err.emit_note(*self.tcx);
                    err
                })?;
            }
        }

        // done
        M::after_stack_push(self)?;
        self.frame_mut().loc = Left(mir::Location::START);

        let span = info_span!("frame", "{}", instance);
        self.frame_mut().tracing_span.enter(span);

        Ok(())
    }

    /// Jump to the given block.
    #[inline]
    pub fn go_to_block(&mut self, target: mir::BasicBlock) {
        self.frame_mut().loc = Left(mir::Location { block: target, statement_index: 0 });
    }

    /// *Return* to the given `target` basic block.
    /// Do *not* use for unwinding! Use `unwind_to_block` instead.
    ///
    /// If `target` is `None`, that indicates the function cannot return, so we raise UB.
    pub fn return_to_block(&mut self, target: Option<mir::BasicBlock>) -> InterpResult<'tcx> {
        if let Some(target) = target {
            self.go_to_block(target);
            Ok(())
        } else {
            throw_ub!(Unreachable)
        }
    }

    /// *Unwind* to the given `target` basic block.
    /// Do *not* use for returning! Use `return_to_block` instead.
    ///
    /// If `target` is `UnwindAction::Continue`, that indicates the function does not need cleanup
    /// during unwinding, and we will just keep propagating that upwards.
    ///
    /// If `target` is `UnwindAction::Unreachable`, that indicates the function does not allow
    /// unwinding, and doing so is UB.
    #[cold] // usually we have normal returns, not unwinding
    pub fn unwind_to_block(&mut self, target: mir::UnwindAction) -> InterpResult<'tcx> {
        self.frame_mut().loc = match target {
            mir::UnwindAction::Cleanup(block) => Left(mir::Location { block, statement_index: 0 }),
            mir::UnwindAction::Continue => Right(self.frame_mut().body.span),
            mir::UnwindAction::Unreachable => {
                throw_ub_custom!(fluent::const_eval_unreachable_unwind);
            }
            mir::UnwindAction::Terminate(reason) => {
                self.frame_mut().loc = Right(self.frame_mut().body.span);
                M::unwind_terminate(self, reason)?;
                // This might have pushed a new stack frame, or it terminated execution.
                // Either way, `loc` will not be updated.
                return Ok(());
            }
        };
        Ok(())
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
    #[instrument(skip(self), level = "debug")]
    pub(super) fn pop_stack_frame(&mut self, unwinding: bool) -> InterpResult<'tcx> {
        info!(
            "popping stack frame ({})",
            if unwinding { "during unwinding" } else { "returning from function" }
        );

        // Check `unwinding`.
        assert_eq!(
            unwinding,
            match self.frame().loc {
                Left(loc) => self.body().basic_blocks[loc.block].is_cleanup,
                Right(_) => true,
            }
        );
        if unwinding && self.frame_idx() == 0 {
            throw_ub_custom!(fluent::const_eval_unwind_past_top);
        }

        M::before_stack_pop(self, self.frame())?;

        // Copy return value. Must of course happen *before* we deallocate the locals.
        let copy_ret_result = if !unwinding {
            let op = self
                .local_to_op(self.frame(), mir::RETURN_PLACE, None)
                .expect("return place should always be live");
            let dest = self.frame().return_place.clone();
            let err = if self.stack().len() == 1 {
                // The initializer of constants and statics will get validated separately
                // after the constant has been fully evaluated. While we could fall back to the default
                // code path, that will cause -Zenforce-validity to cycle on static initializers.
                // Reading from a static's memory is not allowed during its evaluation, and will always
                // trigger a cycle error. Validation must read from the memory of the current item.
                // For Miri this means we do not validate the root frame return value,
                // but Miri anyway calls `read_target_isize` on that so separate validation
                // is not needed.
                self.copy_op_no_dest_validation(&op, &dest)
            } else {
                self.copy_op_allow_transmute(&op, &dest)
            };
            trace!("return value: {:?}", self.dump_place(&dest.into()));
            // We delay actually short-circuiting on this error until *after* the stack frame is
            // popped, since we want this error to be attributed to the caller, whose type defines
            // this transmute.
            err
        } else {
            Ok(())
        };

        // Cleanup: deallocate locals.
        // Usually we want to clean up (deallocate locals), but in a few rare cases we don't.
        // We do this while the frame is still on the stack, so errors point to the callee.
        let return_to_block = self.frame().return_to_block;
        let cleanup = match return_to_block {
            StackPopCleanup::Goto { .. } => true,
            StackPopCleanup::Root { cleanup, .. } => cleanup,
        };
        if cleanup {
            // We need to take the locals out, since we need to mutate while iterating.
            let locals = mem::take(&mut self.frame_mut().locals);
            for local in &locals {
                self.deallocate_local(local.value)?;
            }
        }

        // All right, now it is time to actually pop the frame.
        // Note that its locals are gone already, but that's fine.
        let frame =
            self.stack_mut().pop().expect("tried to pop a stack frame, but there were none");
        // Report error from return value copy, if any.
        copy_ret_result?;

        // If we are not doing cleanup, also skip everything else.
        if !cleanup {
            assert!(self.stack().is_empty(), "only the topmost frame should ever be leaked");
            assert!(!unwinding, "tried to skip cleanup during unwinding");
            // Skip machine hook.
            return Ok(());
        }
        if M::after_stack_pop(self, frame, unwinding)? == StackPopJump::NoJump {
            // The hook already did everything.
            return Ok(());
        }

        // Normal return, figure out where to jump.
        if unwinding {
            // Follow the unwind edge.
            let unwind = match return_to_block {
                StackPopCleanup::Goto { unwind, .. } => unwind,
                StackPopCleanup::Root { .. } => {
                    panic!("encountered StackPopCleanup::Root when unwinding!")
                }
            };
            // This must be the very last thing that happens, since it can in fact push a new stack frame.
            self.unwind_to_block(unwind)
        } else {
            // Follow the normal return edge.
            match return_to_block {
                StackPopCleanup::Goto { ret, .. } => self.return_to_block(ret),
                StackPopCleanup::Root { .. } => {
                    assert!(
                        self.stack().is_empty(),
                        "only the topmost frame can have StackPopCleanup::Root"
                    );
                    Ok(())
                }
            }
        }
    }

    /// In the current stack frame, mark all locals as live that are not arguments and don't have
    /// `Storage*` annotations (this includes the return place).
    pub fn storage_live_for_always_live_locals(&mut self) -> InterpResult<'tcx> {
        self.storage_live(mir::RETURN_PLACE)?;

        let body = self.body();
        let always_live = always_storage_live_locals(body);
        for local in body.vars_and_temps_iter() {
            if always_live.contains(local) {
                self.storage_live(local)?;
            }
        }
        Ok(())
    }

    pub fn storage_live_dyn(
        &mut self,
        local: mir::Local,
        meta: MemPlaceMeta<M::Provenance>,
    ) -> InterpResult<'tcx> {
        trace!("{:?} is now live", local);

        // We avoid `ty.is_trivially_sized` since that (a) cannot assume WF, so it recurses through
        // all fields of a tuple, and (b) does something expensive for ADTs.
        fn is_very_trivially_sized(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
                | ty::Uint(_)
                | ty::Int(_)
                | ty::Bool
                | ty::Float(_)
                | ty::FnDef(..)
                | ty::FnPtr(_)
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

                ty::Str | ty::Slice(_) | ty::Dynamic(..) | ty::Foreign(..) => false,

                ty::Tuple(tys) => tys.last().iter().all(|ty| is_very_trivially_sized(**ty)),

                // We don't want to do any queries, so there is not much we can do with ADTs.
                ty::Adt(..) => false,

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
            assert!(!meta.has_meta()); // we're dropping the metadata
            // Just make this an efficient immediate.
            // Note that not calling `layout_of` here does have one real consequence:
            // if the type is too big, we'll only notice this when the local is actually initialized,
            // which is a bit too late -- we should ideally notice this already here, when the memory
            // is conceptually allocated. But given how rare that error is and that this is a hot function,
            // we accept this downside for now.
            Operand::Immediate(Immediate::Uninit)
        });

        // StorageLive expects the local to be dead, and marks it live.
        let old = mem::replace(&mut self.frame_mut().locals[local].value, local_val);
        if !matches!(old, LocalValue::Dead) {
            throw_ub_custom!(fluent::const_eval_double_storage_live);
        }
        Ok(())
    }

    /// Mark a storage as live, killing the previous content.
    #[inline(always)]
    pub fn storage_live(&mut self, local: mir::Local) -> InterpResult<'tcx> {
        self.storage_live_dyn(local, MemPlaceMeta::None)
    }

    pub fn storage_dead(&mut self, local: mir::Local) -> InterpResult<'tcx> {
        assert!(local != mir::RETURN_PLACE, "Cannot make return place dead");
        trace!("{:?} is now dead", local);

        // It is entirely okay for this local to be already dead (at least that's how we currently generate MIR)
        let old = mem::replace(&mut self.frame_mut().locals[local].value, LocalValue::Dead);
        self.deallocate_local(old)?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
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
        Ok(())
    }

    /// Call a query that can return `ErrorHandled`. Should be used for statics and other globals.
    /// (`mir::Const`/`ty::Const` have `eval` methods that can be used directly instead.)
    pub fn ctfe_query<T>(
        &self,
        query: impl FnOnce(TyCtxtAt<'tcx>) -> Result<T, ErrorHandled>,
    ) -> Result<T, ErrorHandled> {
        // Use a precise span for better cycle errors.
        query(self.tcx.at(self.cur_span())).map_err(|err| {
            err.emit_note(*self.tcx);
            err
        })
    }

    pub fn eval_global(
        &self,
        instance: ty::Instance<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        let gid = GlobalId { instance, promoted: None };
        let val = if self.tcx.is_static(gid.instance.def_id()) {
            let alloc_id = self.tcx.reserve_and_set_static_alloc(gid.instance.def_id());

            let ty = instance.ty(self.tcx.tcx, self.param_env);
            mir::ConstAlloc { alloc_id, ty }
        } else {
            self.ctfe_query(|tcx| tcx.eval_to_allocation_raw(self.param_env.and(gid)))?
        };
        self.raw_const_to_mplace(val)
    }

    pub fn eval_mir_constant(
        &self,
        val: &mir::Const<'tcx>,
        span: Option<Span>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        M::eval_mir_constant(self, *val, span, layout, |ecx, val, span, layout| {
            let const_val = val.eval(*ecx.tcx, ecx.param_env, span).map_err(|err| {
                // FIXME: somehow this is reachable even when POST_MONO_CHECKS is on.
                // Are we not always populating `required_consts`?
                err.emit_note(*ecx.tcx);
                err
            })?;
            ecx.const_val_to_op(const_val, val.ty(), layout)
        })
    }

    #[must_use]
    pub fn dump_place(
        &self,
        place: &PlaceTy<'tcx, M::Provenance>,
    ) -> PlacePrinter<'_, 'mir, 'tcx, M> {
        PlacePrinter { ecx: self, place: *place.place() }
    }

    #[must_use]
    pub fn generate_stacktrace_from_stack(
        stack: &[Frame<'mir, 'tcx, M::Provenance, M::FrameExtra>],
    ) -> Vec<FrameInfo<'tcx>> {
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

    #[must_use]
    pub fn generate_stacktrace(&self) -> Vec<FrameInfo<'tcx>> {
        Self::generate_stacktrace_from_stack(self.stack())
    }
}

#[doc(hidden)]
/// Helper struct for the `dump_place` function.
pub struct PlacePrinter<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    ecx: &'a InterpCx<'mir, 'tcx, M>,
    place: Place<M::Provenance>,
}

impl<'a, 'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> std::fmt::Debug
    for PlacePrinter<'a, 'mir, 'tcx, M>
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.place {
            Place::Local { frame, local, offset } => {
                let mut allocs = Vec::new();
                write!(fmt, "{local:?}")?;
                if let Some(offset) = offset {
                    write!(fmt, "+{:#x}", offset.bytes())?;
                }
                if frame != self.ecx.frame_idx() {
                    write!(fmt, " ({} frames up)", self.ecx.frame_idx() - frame)?;
                }
                write!(fmt, ":")?;

                match self.ecx.stack()[frame].locals[local].value {
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

                write!(fmt, ": {:?}", self.ecx.dump_allocs(allocs.into_iter().flatten().collect()))
            }
            Place::Ptr(mplace) => match mplace.ptr.provenance.and_then(Provenance::get_alloc_id) {
                Some(alloc_id) => {
                    write!(fmt, "by ref {:?}: {:?}", mplace.ptr, self.ecx.dump_alloc(alloc_id))
                }
                ptr => write!(fmt, " integral by ref: {ptr:?}"),
            },
        }
    }
}
