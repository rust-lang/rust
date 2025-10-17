use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{AllocInit, Allocation, GlobalAlloc, InterpResult, Pointer};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{TyCtxt, TypeVisitable, TypeVisitableExt};
use tracing::debug;

use super::{InterpCx, MPlaceTy, MemoryKind, interp_ok, throw_inval};
use crate::const_eval::{CompileTimeInterpCx, CompileTimeMachine, InterpretationResult};

/// Checks whether a type contains generic parameters which must be instantiated.
///
/// In case it does, returns a `TooGeneric` const eval error.
pub(crate) fn ensure_monomorphic_enough<'tcx, T>(_tcx: TyCtxt<'tcx>, ty: T) -> InterpResult<'tcx>
where
    T: TypeVisitable<TyCtxt<'tcx>>,
{
    debug!("ensure_monomorphic_enough: ty={:?}", ty);
    if ty.has_param() {
        throw_inval!(TooGeneric);
    }
    interp_ok(())
}

impl<'tcx> InterpretationResult<'tcx> for mir::interpret::ConstAllocation<'tcx> {
    fn make_result(
        mplace: MPlaceTy<'tcx>,
        ecx: &mut InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    ) -> Self {
        let alloc_id = mplace.ptr().provenance.unwrap().alloc_id();
        let alloc = ecx.memory.alloc_map.swap_remove(&alloc_id).unwrap().1;
        ecx.tcx.mk_const_alloc(alloc)
    }
}

pub(crate) fn create_static_alloc<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    static_def_id: LocalDefId,
    layout: TyAndLayout<'tcx>,
) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
    // Inherit size and align from the `GlobalAlloc::Static` so we can avoid duplicating
    // the alignment attribute logic.
    let (size, align) =
        GlobalAlloc::Static(static_def_id.into()).size_and_align(*ecx.tcx, ecx.typing_env);
    assert_eq!(size, layout.size);
    assert!(align >= layout.align.abi);

    let alloc = Allocation::try_new(size, align, AllocInit::Uninit, ())?;
    let alloc_id = ecx.tcx.reserve_and_set_static_alloc(static_def_id.into());
    assert_eq!(ecx.machine.static_root_ids, None);
    ecx.machine.static_root_ids = Some((alloc_id, static_def_id));
    assert!(ecx.memory.alloc_map.insert(alloc_id, (MemoryKind::Stack, alloc)).is_none());
    interp_ok(ecx.ptr_to_mplace(Pointer::from(alloc_id).into(), layout))
}

/// A marker trait returned by [crate::interpret::Machine::enter_trace_span], identifying either a
/// real [tracing::span::EnteredSpan] in case tracing is enabled, or the dummy type `()` when
/// tracing is disabled. Also see [crate::enter_trace_span!] below.
pub trait EnteredTraceSpan {
    /// Allows executing an alternative function when tracing is disabled. Useful for example if you
    /// want to open a trace span when tracing is enabled, and alternatively just log a line when
    /// tracing is disabled.
    fn or_if_tracing_disabled(self, f: impl FnOnce()) -> Self;
}
impl EnteredTraceSpan for () {
    fn or_if_tracing_disabled(self, f: impl FnOnce()) -> Self {
        f(); // tracing is disabled, execute the function
        self
    }
}
impl EnteredTraceSpan for tracing::span::EnteredSpan {
    fn or_if_tracing_disabled(self, _f: impl FnOnce()) -> Self {
        self // tracing is enabled, don't execute anything
    }
}

/// Shortand for calling [crate::interpret::Machine::enter_trace_span] on a [tracing::info_span!].
/// This is supposed to be compiled out when [crate::interpret::Machine::enter_trace_span] has the
/// default implementation (i.e. when it does not actually enter the span but instead returns `()`).
/// This macro takes a type implementing the [crate::interpret::Machine] trait as its first argument
/// and otherwise accepts the same syntax as [tracing::span!] (see some tips below).
/// Note: the result of this macro **must be used** because the span is exited when it's dropped.
///
/// ### Syntax accepted by this macro
///
/// The full documentation for the [tracing::span!] syntax can be found at [tracing] under "Using the
/// Macros". A few possibly confusing syntaxes are listed here:
/// ```rust
/// # use rustc_const_eval::enter_trace_span;
/// # type M = rustc_const_eval::const_eval::CompileTimeMachine<'static>;
/// # let my_display_var = String::new();
/// # let my_debug_var = String::new();
/// // logs a span named "hello" with a field named "arg" of value 42 (works only because
/// // 42 implements the tracing::Value trait, otherwise use one of the options below)
/// let _trace = enter_trace_span!(M, "hello", arg = 42);
/// // logs a field called "my_display_var" using the Display implementation
/// let _trace = enter_trace_span!(M, "hello", %my_display_var);
/// // logs a field called "my_debug_var" using the Debug implementation
/// let _trace = enter_trace_span!(M, "hello", ?my_debug_var);
///  ```
///
/// ### `NAME::SUBNAME` syntax
///
/// In addition to the syntax accepted by [tracing::span!], this macro optionally allows passing
/// the span name (i.e. the first macro argument) in the form `NAME::SUBNAME` (without quotes) to
/// indicate that the span has name "NAME" (usually the name of the component) and has an additional
/// more specific name "SUBNAME" (usually the function name). The latter is passed to the [tracing]
/// infrastructure as a span field with the name "NAME". This allows not being distracted by
/// subnames when looking at the trace in <https://ui.perfetto.dev>, but when deeper introspection
/// is needed within a component, it's still possible to view the subnames directly in the UI by
/// selecting a span, clicking on the "NAME" argument on the right, and clicking on "Visualize
/// argument values".
/// ```rust
/// # use rustc_const_eval::enter_trace_span;
/// # type M = rustc_const_eval::const_eval::CompileTimeMachine<'static>;
/// // for example, the first will expand to the second
/// let _trace = enter_trace_span!(M, borrow_tracker::on_stack_pop, /* ... */);
/// let _trace = enter_trace_span!(M, "borrow_tracker", borrow_tracker = "on_stack_pop", /* ... */);
/// ```
///
/// ### `tracing_separate_thread` parameter
///
/// This macro was introduced to obtain better traces of Miri without impacting release performance.
/// Miri saves traces using the the `tracing_chrome` `tracing::Layer` so that they can be visualized
/// in <https://ui.perfetto.dev>. To instruct `tracing_chrome` to put some spans on a separate trace
/// thread/line than other spans when viewed in <https://ui.perfetto.dev>, you can pass
/// `tracing_separate_thread = tracing::field::Empty` to the tracing macros. This is useful to
/// separate out spans which just indicate the current step or program frame being processed by the
/// interpreter. You should use a value of [tracing::field::Empty] so that other tracing layers
/// (e.g. the logger) will ignore the `tracing_separate_thread` field. For example:
/// ```rust
/// # use rustc_const_eval::enter_trace_span;
/// # type M = rustc_const_eval::const_eval::CompileTimeMachine<'static>;
/// let _trace = enter_trace_span!(M, step::eval_statement, tracing_separate_thread = tracing::field::Empty);
/// ```
///
/// ### Executing something else when tracing is disabled
///
/// [crate::interpret::Machine::enter_trace_span] returns [EnteredTraceSpan], on which you can call
/// [EnteredTraceSpan::or_if_tracing_disabled], to e.g. log a line as an alternative to the tracing
/// span for when tracing is disabled. For example:
/// ```rust
/// # use rustc_const_eval::enter_trace_span;
/// # use rustc_const_eval::interpret::EnteredTraceSpan;
/// # type M = rustc_const_eval::const_eval::CompileTimeMachine<'static>;
/// let _trace = enter_trace_span!(M, step::eval_statement)
///     .or_if_tracing_disabled(|| tracing::info!("eval_statement"));
/// ```
#[macro_export]
macro_rules! enter_trace_span {
    ($machine:ty, $name:ident :: $subname:ident $($tt:tt)*) => {
        $crate::enter_trace_span!($machine, stringify!($name), $name = %stringify!($subname) $($tt)*)
    };

    ($machine:ty, $($tt:tt)*) => {
        <$machine as $crate::interpret::Machine>::enter_trace_span(|| tracing::info_span!($($tt)*))
    };
}
