use std::assert_matches::debug_assert_matches;

use either::{Left, Right};
use rustc_abi::{Align, HasDataLayout, Size, TargetDataLayout};
use rustc_errors::DiagCtxtHandle;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{ErrorHandled, InvalidMetaKind, ReportedErrorInfo};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::{
    self, FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeFoldable, TypingEnv, Variance};
use rustc_middle::{mir, span_bug};
use rustc_session::Limit;
use rustc_span::Span;
use rustc_target::callconv::FnAbi;
use tracing::{debug, trace};

use super::{
    Frame, FrameInfo, GlobalId, InterpErrorInfo, InterpErrorKind, InterpResult, MPlaceTy, Machine,
    MemPlaceMeta, Memory, OpTy, Place, PlaceTy, PointerArithmetic, Projectable, Provenance,
    err_inval, interp_ok, throw_inval, throw_ub, throw_ub_custom,
};
use crate::{ReportErrorExt, fluent_generated as fluent, util};

pub struct InterpCx<'tcx, M: Machine<'tcx>> {
    /// Stores the `Machine` instance.
    ///
    /// Note: the stack is provided by the machine.
    pub machine: M,

    /// The results of the type checker, from rustc.
    /// The span in this is the "root" of the evaluation, i.e., the const
    /// we are evaluating (if this is CTFE).
    pub tcx: TyCtxtAt<'tcx>,

    /// The current context in case we're evaluating in a
    /// polymorphic context. This always uses `ty::TypingMode::PostAnalysis`.
    pub(super) typing_env: ty::TypingEnv<'tcx>,

    /// The virtual memory system.
    pub memory: Memory<'tcx, M>,

    /// The recursion limit (cached from `tcx.recursion_limit(())`)
    pub recursion_limit: Limit,
}

impl<'tcx, M: Machine<'tcx>> HasDataLayout for InterpCx<'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx, M> layout::HasTyCtxt<'tcx> for InterpCx<'tcx, M>
where
    M: Machine<'tcx>,
{
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        *self.tcx
    }
}

impl<'tcx, M> layout::HasTypingEnv<'tcx> for InterpCx<'tcx, M>
where
    M: Machine<'tcx>,
{
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }
}

impl<'tcx, M: Machine<'tcx>> LayoutOfHelpers<'tcx> for InterpCx<'tcx, M> {
    type LayoutOfResult = Result<TyAndLayout<'tcx>, InterpErrorKind<'tcx>>;

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
    ) -> InterpErrorKind<'tcx> {
        err_inval!(Layout(err))
    }
}

impl<'tcx, M: Machine<'tcx>> FnAbiOfHelpers<'tcx> for InterpCx<'tcx, M> {
    type FnAbiOfResult = Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, InterpErrorKind<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        _span: Span,
        _fn_abi_request: FnAbiRequest<'tcx>,
    ) -> InterpErrorKind<'tcx> {
        match err {
            FnAbiError::Layout(err) => err_inval!(Layout(err)),
        }
    }
}

/// Test if it is valid for a MIR assignment to assign `src`-typed place to `dest`-typed value.
/// This test should be symmetric, as it is primarily about layout compatibility.
pub(super) fn mir_assign_valid_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    src: TyAndLayout<'tcx>,
    dest: TyAndLayout<'tcx>,
) -> bool {
    // Type-changing assignments can happen when subtyping is used. While
    // all normal lifetimes are erased, higher-ranked types with their
    // late-bound lifetimes are still around and can lead to type
    // differences.
    if util::relate_types(tcx, typing_env, Variance::Covariant, src.ty, dest.ty) {
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
    typing_env: TypingEnv<'tcx>,
    known_layout: Option<TyAndLayout<'tcx>>,
    compute: impl FnOnce() -> InterpResult<'tcx, TyAndLayout<'tcx>>,
) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
    match known_layout {
        None => compute(),
        Some(known_layout) => {
            if cfg!(debug_assertions) {
                let check_layout = compute()?;
                if !mir_assign_valid_types(tcx.tcx, typing_env, check_layout, known_layout) {
                    span_bug!(
                        tcx.span,
                        "expected type differs from actual type.\nexpected: {}\nactual: {}",
                        known_layout.ty,
                        check_layout.ty,
                    );
                }
            }
            interp_ok(known_layout)
        }
    }
}

/// Turn the given error into a human-readable string. Expects the string to be printed, so if
/// `RUSTC_CTFE_BACKTRACE` is set this will show a backtrace of the rustc internals that
/// triggered the error.
///
/// This is NOT the preferred way to render an error; use `report` from `const_eval` instead.
/// However, this is useful when error messages appear in ICEs.
pub fn format_interp_error<'tcx>(dcx: DiagCtxtHandle<'_>, e: InterpErrorInfo<'tcx>) -> String {
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

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        root_span: Span,
        typing_env: ty::TypingEnv<'tcx>,
        machine: M,
    ) -> Self {
        // Const eval always happens in post analysis mode in order to be able to use the hidden types of
        // opaque types. This is needed for trivial things like `size_of`, but also for using associated
        // types that are not specified in the opaque type. We also use MIR bodies whose opaque types have
        // already been revealed, so we'd be able to at least partially observe the hidden types anyways.
        debug_assert_matches!(typing_env.typing_mode, ty::TypingMode::PostAnalysis);
        InterpCx {
            machine,
            tcx: tcx.at(root_span),
            typing_env,
            memory: Memory::new(),
            recursion_limit: tcx.recursion_limit(),
        }
    }

    /// Returns the span of the currently executed statement/terminator.
    /// This is the span typically used for error reporting.
    #[inline(always)]
    pub fn cur_span(&self) -> Span {
        // This deliberately does *not* honor `requires_caller_location` since it is used for much
        // more than just panics.
        self.stack().last().map_or(self.tcx.span, |f| f.current_span())
    }

    pub(crate) fn stack(&self) -> &[Frame<'tcx, M::Provenance, M::FrameExtra>] {
        M::stack(self)
    }

    #[inline(always)]
    pub(crate) fn stack_mut(&mut self) -> &mut Vec<Frame<'tcx, M::Provenance, M::FrameExtra>> {
        M::stack_mut(self)
    }

    #[inline(always)]
    pub fn frame_idx(&self) -> usize {
        let stack = self.stack();
        assert!(!stack.is_empty());
        stack.len() - 1
    }

    #[inline(always)]
    pub fn frame(&self) -> &Frame<'tcx, M::Provenance, M::FrameExtra> {
        self.stack().last().expect("no call frames exist")
    }

    #[inline(always)]
    pub fn frame_mut(&mut self) -> &mut Frame<'tcx, M::Provenance, M::FrameExtra> {
        self.stack_mut().last_mut().expect("no call frames exist")
    }

    #[inline(always)]
    pub fn body(&self) -> &'tcx mir::Body<'tcx> {
        self.frame().body
    }

    #[inline]
    pub fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        ty.is_freeze(*self.tcx, self.typing_env)
    }

    pub fn load_mir(
        &self,
        instance: ty::InstanceKind<'tcx>,
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
            throw_inval!(AlreadyReported(ReportedErrorInfo::non_const_eval_error(err)));
        }
        interp_ok(body)
    }

    /// Call this on things you got out of the MIR (so it is as generic as the current
    /// stack frame), to bring it into the proper environment for this interpreter.
    pub fn instantiate_from_current_frame_and_normalize_erasing_regions<
        T: TypeFoldable<TyCtxt<'tcx>>,
    >(
        &self,
        value: T,
    ) -> Result<T, ErrorHandled> {
        self.instantiate_from_frame_and_normalize_erasing_regions(self.frame(), value)
    }

    /// Call this on things you got out of the MIR (so it is as generic as the provided
    /// stack frame), to bring it into the proper environment for this interpreter.
    pub fn instantiate_from_frame_and_normalize_erasing_regions<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        frame: &Frame<'tcx, M::Provenance, M::FrameExtra>,
        value: T,
    ) -> Result<T, ErrorHandled> {
        frame
            .instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                *self.tcx,
                self.typing_env,
                ty::EarlyBinder::bind(value),
            )
            .map_err(|_| ErrorHandled::TooGeneric(self.cur_span()))
    }

    /// The `args` are assumed to already be in our interpreter "universe".
    pub(super) fn resolve(
        &self,
        def: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> InterpResult<'tcx, ty::Instance<'tcx>> {
        trace!("resolve: {:?}, {:#?}", def, args);
        trace!("typing_env: {:#?}", self.typing_env);
        trace!("args: {:#?}", args);
        match ty::Instance::try_resolve(*self.tcx, self.typing_env, def, args) {
            Ok(Some(instance)) => interp_ok(instance),
            Ok(None) => throw_inval!(TooGeneric),

            // FIXME(eddyb) this could be a bit more specific than `AlreadyReported`.
            Err(error_guaranteed) => throw_inval!(AlreadyReported(
                ReportedErrorInfo::non_const_eval_error(error_guaranteed)
            )),
        }
    }

    /// Walks up the callstack from the intrinsic's callsite, searching for the first callsite in a
    /// frame which is not `#[track_caller]`. This matches the `caller_location` intrinsic,
    /// and is primarily intended for the panic machinery.
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

    /// Returns the actual dynamic size and alignment of the place at the given type.
    /// Only the "meta" (metadata) part of the place matters.
    /// This can fail to provide an answer for extern types.
    pub(super) fn size_and_align_of(
        &self,
        metadata: &MemPlaceMeta<M::Provenance>,
        layout: &TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, Option<(Size, Align)>> {
        if layout.is_sized() {
            return interp_ok(Some((layout.size, layout.align.abi)));
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
                    return interp_ok(None);
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
                interp_ok(Some((full_size, full_align)))
            }
            ty::Dynamic(expected_trait, _, ty::Dyn) => {
                let vtable = metadata.unwrap_meta().to_pointer(self)?;
                // Read size and align from vtable (already checks size).
                interp_ok(Some(self.get_vtable_size_and_align(vtable, Some(expected_trait))?))
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
                interp_ok(Some((size, elem.align.abi)))
            }

            ty::Foreign(_) => interp_ok(None),

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
            interp_ok(())
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
                return interp_ok(());
            }
        };
        interp_ok(())
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

            let ty = instance.ty(self.tcx.tcx, self.typing_env);
            mir::ConstAlloc { alloc_id, ty }
        } else {
            self.ctfe_query(|tcx| tcx.eval_to_allocation_raw(self.typing_env.as_query_input(gid)))?
        };
        self.raw_const_to_mplace(val)
    }

    pub fn eval_mir_constant(
        &self,
        val: &mir::Const<'tcx>,
        span: Span,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        M::eval_mir_constant(self, *val, span, layout, |ecx, val, span, layout| {
            let const_val = val.eval(*ecx.tcx, ecx.typing_env, span).map_err(|err| {
                if M::ALL_CONSTS_ARE_PRECHECKED {
                    match err {
                        ErrorHandled::TooGeneric(..) => {},
                        ErrorHandled::Reported(reported, span) => {
                            if reported.is_allowed_in_infallible() {
                                // These errors can just sometimes happen, even when the expression
                                // is nominally "infallible", e.g. when running out of memory
                                // or when some layout could not be computed.
                            } else {
                                // Looks like the const is not captured by `required_consts`, that's bad.
                                span_bug!(span, "interpret const eval failure of {val:?} which is not in required_consts");
                            }
                        }
                    }
                }
                err.emit_note(*ecx.tcx);
                err
            })?;
            ecx.const_val_to_op(const_val, val.ty(), layout)
        })
    }

    #[must_use]
    pub fn dump_place(&self, place: &PlaceTy<'tcx, M::Provenance>) -> PlacePrinter<'_, 'tcx, M> {
        PlacePrinter { ecx: self, place: *place.place() }
    }

    #[must_use]
    pub fn generate_stacktrace(&self) -> Vec<FrameInfo<'tcx>> {
        Frame::generate_stacktrace_from_stack(self.stack())
    }

    pub fn adjust_nan<F1, F2>(&self, f: F2, inputs: &[F1]) -> F2
    where
        F1: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F2>,
        F2: rustc_apfloat::Float,
    {
        if f.is_nan() { M::generate_nan(self, inputs) } else { f }
    }
}

#[doc(hidden)]
/// Helper struct for the `dump_place` function.
pub struct PlacePrinter<'a, 'tcx, M: Machine<'tcx>> {
    ecx: &'a InterpCx<'tcx, M>,
    place: Place<M::Provenance>,
}

impl<'a, 'tcx, M: Machine<'tcx>> std::fmt::Debug for PlacePrinter<'a, 'tcx, M> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.place {
            Place::Local { local, offset, locals_addr } => {
                debug_assert_eq!(locals_addr, self.ecx.frame().locals_addr());
                let mut allocs = Vec::new();
                write!(fmt, "{local:?}")?;
                if let Some(offset) = offset {
                    write!(fmt, "+{:#x}", offset.bytes())?;
                }
                write!(fmt, ":")?;

                self.ecx.frame().locals[local].print(&mut allocs, fmt)?;

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
