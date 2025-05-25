use rustc_middle::mir::interpret::{AllocId, ConstAllocation, InterpResult};
use rustc_middle::mir::*;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::{bug, span_bug, ty};
use rustc_span::def_id::DefId;
use rustc_target::callconv::FnAbi;

use crate::interpret::{
    self, HasStaticRootDefId, ImmTy, Immediate, InterpCx, PointerArithmetic, interp_ok,
    throw_machine_stop,
};

/// Macro for machine-specific `InterpError` without allocation.
/// (These will never be shown to the user, but they help diagnose ICEs.)
pub macro throw_machine_stop_str($($tt:tt)*) {{
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
        fn diagnostic_message(&self) -> rustc_errors::DiagMessage {
            self.to_string().into()
        }

        fn add_args(
            self: Box<Self>,
            _: &mut dyn FnMut(rustc_errors::DiagArgName, rustc_errors::DiagArgValue),
        ) {}
    }
    throw_machine_stop!(Zst)
}}

pub struct DummyMachine;

impl HasStaticRootDefId for DummyMachine {
    fn static_def_id(&self) -> Option<rustc_hir::def_id::LocalDefId> {
        None
    }
}

impl<'tcx> interpret::Machine<'tcx> for DummyMachine {
    interpret::compile_time_machine!(<'tcx>);
    type MemoryKind = !;
    const PANIC_ON_ALLOC_FAIL: bool = true;

    // We want to just eval random consts in the program, so `eval_mir_const` can fail.
    const ALL_CONSTS_ARE_PRECHECKED: bool = false;

    #[inline(always)]
    fn enforce_alignment(_ecx: &InterpCx<'tcx, Self>) -> bool {
        false // no reason to enforce alignment
    }

    fn enforce_validity(_ecx: &InterpCx<'tcx, Self>, _layout: TyAndLayout<'tcx>) -> bool {
        false
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

        interp_ok(())
    }

    fn find_mir_or_eval_fn(
        _ecx: &mut InterpCx<'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _abi: &FnAbi<'tcx, Ty<'tcx>>,
        _args: &[interpret::FnArg<'tcx, Self::Provenance>],
        _destination: &interpret::PlaceTy<'tcx, Self::Provenance>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx, Option<(&'tcx Body<'tcx>, ty::Instance<'tcx>)>> {
        unimplemented!()
    }

    fn panic_nounwind(
        _ecx: &mut InterpCx<'tcx, Self>,
        _msg: &str,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn call_intrinsic(
        _ecx: &mut InterpCx<'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[interpret::OpTy<'tcx, Self::Provenance>],
        _destination: &interpret::PlaceTy<'tcx, Self::Provenance>,
        _target: Option<BasicBlock>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        unimplemented!()
    }

    fn assert_panic(
        _ecx: &mut InterpCx<'tcx, Self>,
        _msg: &rustc_middle::mir::AssertMessage<'tcx>,
        _unwind: UnwindAction,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn binary_ptr_op(
        ecx: &InterpCx<'tcx, Self>,
        bin_op: BinOp,
        left: &interpret::ImmTy<'tcx, Self::Provenance>,
        right: &interpret::ImmTy<'tcx, Self::Provenance>,
    ) -> interpret::InterpResult<'tcx, ImmTy<'tcx, Self::Provenance>> {
        use rustc_middle::mir::BinOp::*;
        interp_ok(match bin_op {
            Eq | Ne | Lt | Le | Gt | Ge => {
                // Types can differ, e.g. fn ptrs with different `for`.
                assert_eq!(left.layout.backend_repr, right.layout.backend_repr);
                let size = ecx.pointer_size();
                // Just compare the bits. ScalarPairs are compared lexicographically.
                // We thus always compare pairs and simply fill scalars up with 0.
                // If the pointer has provenance, `to_bits` will return `Err` and we bail out.
                let left = match **left {
                    Immediate::Scalar(l) => (l.to_bits(size)?, 0),
                    Immediate::ScalarPair(l1, l2) => (l1.to_bits(size)?, l2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let right = match **right {
                    Immediate::Scalar(r) => (r.to_bits(size)?, 0),
                    Immediate::ScalarPair(r1, r2) => (r1.to_bits(size)?, r2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let res = match bin_op {
                    Eq => left == right,
                    Ne => left != right,
                    Lt => left < right,
                    Le => left <= right,
                    Gt => left > right,
                    Ge => left >= right,
                    _ => bug!(),
                };
                ImmTy::from_bool(res, *ecx.tcx)
            }

            // Some more operations are possible with atomics.
            // The return value always has the provenance of the *left* operand.
            Add | Sub | BitOr | BitAnd | BitXor => {
                throw_machine_stop_str!("pointer arithmetic is not handled")
            }

            _ => span_bug!(ecx.cur_span(), "Invalid operator on pointers: {:?}", bin_op),
        })
    }

    fn expose_provenance(
        _ecx: &InterpCx<'tcx, Self>,
        _provenance: Self::Provenance,
    ) -> interpret::InterpResult<'tcx> {
        unimplemented!()
    }

    fn init_frame(
        _ecx: &mut InterpCx<'tcx, Self>,
        _frame: interpret::Frame<'tcx, Self::Provenance>,
    ) -> interpret::InterpResult<'tcx, interpret::Frame<'tcx, Self::Provenance, Self::FrameExtra>>
    {
        unimplemented!()
    }

    fn stack<'a>(
        _ecx: &'a InterpCx<'tcx, Self>,
    ) -> &'a [interpret::Frame<'tcx, Self::Provenance, Self::FrameExtra>] {
        // Return an empty stack instead of panicking, as `cur_span` uses it to evaluate constants.
        &[]
    }

    fn stack_mut<'a>(
        _ecx: &'a mut InterpCx<'tcx, Self>,
    ) -> &'a mut Vec<interpret::Frame<'tcx, Self::Provenance, Self::FrameExtra>> {
        unimplemented!()
    }
}
