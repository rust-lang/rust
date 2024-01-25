//! Propagates constants for early reporting of statically known
//! assertion failures

use rustc_const_eval::interpret::{
    self, compile_time_machine, AllocId, ConstAllocation, FnArg, Frame, ImmTy, InterpCx,
    InterpResult, OpTy, PlaceTy, Pointer,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, ParamEnv, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi as CallAbi;

/// The maximum number of bytes that we'll allocate space for a local or the return value.
/// Needed for #66397, because otherwise we eval into large places and that can cause OOM or just
/// Severely regress performance.
const MAX_ALLOC_LIMIT: u64 = 1024;

/// Macro for machine-specific `InterpError` without allocation.
/// (These will never be shown to the user, but they help diagnose ICEs.)
pub(crate) macro throw_machine_stop_str($($tt:tt)*) {{
    // We make a new local type for it. The type itself does not carry any information,
    // but its vtable (for the `MachineStopType` trait) does.
    #[derive(Debug)]
    struct Zst;
    // Printing this type shows the desired string.
    impl std::fmt::Display for Zst {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, $($tt)*)
        }
    }

    impl rustc_middle::mir::interpret::MachineStopType for Zst {
        fn diagnostic_message(&self) -> rustc_errors::DiagnosticMessage {
            self.to_string().into()
        }

        fn add_args(
            self: Box<Self>,
            _: &mut dyn FnMut(std::borrow::Cow<'static, str>, rustc_errors::DiagnosticArgValue<'static>),
        ) {}
    }
    throw_machine_stop!(Zst)
}}

pub(crate) struct ConstPropMachine<'mir, 'tcx> {
    /// The virtual call stack.
    stack: Vec<Frame<'mir, 'tcx>>,
    pub written_only_inside_own_block_locals: FxHashSet<Local>,
    pub can_const_prop: IndexVec<Local, ConstPropMode>,
}

impl ConstPropMachine<'_, '_> {
    pub fn new(can_const_prop: IndexVec<Local, ConstPropMode>) -> Self {
        Self {
            stack: Vec::new(),
            written_only_inside_own_block_locals: Default::default(),
            can_const_prop,
        }
    }
}

impl<'mir, 'tcx> interpret::Machine<'mir, 'tcx> for ConstPropMachine<'mir, 'tcx> {
    compile_time_machine!(<'mir, 'tcx>);

    const PANIC_ON_ALLOC_FAIL: bool = true; // all allocations are small (see `MAX_ALLOC_LIMIT`)

    const POST_MONO_CHECKS: bool = false; // this MIR is still generic!

    type MemoryKind = !;

    #[inline(always)]
    fn enforce_alignment(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        false // no reason to enforce alignment
    }

    #[inline(always)]
    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>, _layout: TyAndLayout<'tcx>) -> bool {
        false // for now, we don't enforce validity
    }

    fn load_mir(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _instance: ty::InstanceDef<'tcx>,
    ) -> InterpResult<'tcx, &'tcx Body<'tcx>> {
        throw_machine_stop_str!("calling functions isn't supported in ConstProp")
    }

    fn panic_nounwind(_ecx: &mut InterpCx<'mir, 'tcx, Self>, _msg: &str) -> InterpResult<'tcx> {
        throw_machine_stop_str!("panicking isn't supported in ConstProp")
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _abi: CallAbi,
        _args: &[FnArg<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'mir Body<'tcx>, ty::Instance<'tcx>)>> {
        Ok(None)
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[OpTy<'tcx>],
        _destination: &PlaceTy<'tcx>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> InterpResult<'tcx> {
        throw_machine_stop_str!("calling intrinsics isn't supported in ConstProp")
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: rustc_middle::mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        bug!("panics terminators are not evaluated in ConstProp")
    }

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: BinOp,
        _left: &ImmTy<'tcx>,
        _right: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx>, bool)> {
        // We can't do this because aliasing of memory can differ between const eval and llvm
        throw_machine_stop_str!("pointer arithmetic or comparisons aren't supported in ConstProp")
    }

    fn before_access_local_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
        frame: usize,
        local: Local,
    ) -> InterpResult<'tcx> {
        assert_eq!(frame, 0);
        match ecx.machine.can_const_prop[local] {
            ConstPropMode::NoPropagation => {
                throw_machine_stop_str!(
                    "tried to write to a local that is marked as not propagatable"
                )
            }
            ConstPropMode::OnlyInsideOwnBlock => {
                ecx.machine.written_only_inside_own_block_locals.insert(local);
            }
            ConstPropMode::FullConstProp => {}
        }
        Ok(())
    }

    fn before_access_global(
        _tcx: TyCtxtAt<'tcx>,
        _machine: &Self,
        _alloc_id: AllocId,
        alloc: ConstAllocation<'tcx>,
        _static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx> {
        if is_write {
            throw_machine_stop_str!("can't write to global");
        }
        // If the static allocation is mutable, then we can't const prop it as its content
        // might be different at runtime.
        if alloc.inner().mutability.is_mut() {
            throw_machine_stop_str!("can't access mutable globals in ConstProp");
        }

        Ok(())
    }

    #[inline(always)]
    fn expose_ptr(_ecx: &mut InterpCx<'mir, 'tcx, Self>, _ptr: Pointer) -> InterpResult<'tcx> {
        throw_machine_stop_str!("exposing pointers isn't supported in ConstProp")
    }

    #[inline(always)]
    fn init_frame_extra(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx>> {
        Ok(frame)
    }

    #[inline(always)]
    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>] {
        &ecx.machine.stack
    }

    #[inline(always)]
    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>> {
        &mut ecx.machine.stack
    }
}

/// The mode that `ConstProp` is allowed to run in for a given `Local`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstPropMode {
    /// The `Local` can be propagated into and reads of this `Local` can also be propagated.
    FullConstProp,
    /// The `Local` can only be propagated into and from its own block.
    OnlyInsideOwnBlock,
    /// The `Local` cannot be part of propagation at all. Any statement
    /// referencing it either for reading or writing will not get propagated.
    NoPropagation,
}

pub struct CanConstProp {
    can_const_prop: IndexVec<Local, ConstPropMode>,
    // False at the beginning. Once set, no more assignments are allowed to that local.
    found_assignment: BitSet<Local>,
}

impl CanConstProp {
    /// Returns true if `local` can be propagated
    pub fn check<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        body: &Body<'tcx>,
    ) -> IndexVec<Local, ConstPropMode> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(ConstPropMode::FullConstProp, &body.local_decls),
            found_assignment: BitSet::new_empty(body.local_decls.len()),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            let ty = body.local_decls[local].ty;
            match tcx.layout_of(param_env.and(ty)) {
                Ok(layout) if layout.size < Size::from_bytes(MAX_ALLOC_LIMIT) => {}
                // Either the layout fails to compute, then we can't use this local anyway
                // or the local is too large, then we don't want to.
                _ => {
                    *val = ConstPropMode::NoPropagation;
                    continue;
                }
            }
        }
        // Consider that arguments are assigned on entry.
        for arg in body.args_iter() {
            cpv.found_assignment.insert(arg);
        }
        cpv.visit_body(body);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_place(&mut self, place: &Place<'tcx>, mut context: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;

        // Dereferencing just read the addess of `place.local`.
        if place.projection.first() == Some(&PlaceElem::Deref) {
            context = NonMutatingUse(NonMutatingUseContext::Copy);
        }

        self.visit_local(place.local, context, loc);
        self.visit_projection(place.as_ref(), context, loc);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::PlaceContext::*;
        match context {
            // These are just stores, where the storing is not propagatable, but there may be later
            // mutations of the same local via `Store`
            | MutatingUse(MutatingUseContext::Call)
            | MutatingUse(MutatingUseContext::AsmOutput)
            | MutatingUse(MutatingUseContext::Deinit)
            // Actual store that can possibly even propagate a value
            | MutatingUse(MutatingUseContext::Store)
            | MutatingUse(MutatingUseContext::SetDiscriminant) => {
                if !self.found_assignment.insert(local) {
                    match &mut self.can_const_prop[local] {
                        // If the local can only get propagated in its own block, then we don't have
                        // to worry about multiple assignments, as we'll nuke the const state at the
                        // end of the block anyway, and inside the block we overwrite previous
                        // states as applicable.
                        ConstPropMode::OnlyInsideOwnBlock => {}
                        ConstPropMode::NoPropagation => {}
                        other @ ConstPropMode::FullConstProp => {
                            trace!(
                                "local {:?} can't be propagated because of multiple assignments. Previous state: {:?}",
                                local, other,
                            );
                            *other = ConstPropMode::OnlyInsideOwnBlock;
                        }
                    }
                }
            }
            // Reading constants is allowed an arbitrary number of times
            NonMutatingUse(NonMutatingUseContext::Copy)
            | NonMutatingUse(NonMutatingUseContext::Move)
            | NonMutatingUse(NonMutatingUseContext::Inspect)
            | NonMutatingUse(NonMutatingUseContext::PlaceMention)
            | NonUse(_) => {}

            // These could be propagated with a smarter analysis or just some careful thinking about
            // whether they'd be fine right now.
            MutatingUse(MutatingUseContext::Yield)
            | MutatingUse(MutatingUseContext::Drop)
            | MutatingUse(MutatingUseContext::Retag)
            // These can't ever be propagated under any scheme, as we can't reason about indirect
            // mutation.
            | NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            | NonMutatingUse(NonMutatingUseContext::FakeBorrow)
            | NonMutatingUse(NonMutatingUseContext::AddressOf)
            | MutatingUse(MutatingUseContext::Borrow)
            | MutatingUse(MutatingUseContext::AddressOf) => {
                trace!("local {:?} can't be propagated because it's used: {:?}", local, context);
                self.can_const_prop[local] = ConstPropMode::NoPropagation;
            }
            MutatingUse(MutatingUseContext::Projection)
            | NonMutatingUse(NonMutatingUseContext::Projection) => bug!("visit_place should not pass {context:?} for {local:?}"),
        }
    }
}
