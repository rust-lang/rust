use genmc_sys::{GenmcHandlerResult, MemOrdering, RMWBinOp};
use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::mir::interpret;
use rustc_middle::ty::ScalarInt;
use rustc_middle::{mir, throw_ub_format};

use super::GenmcScalar;
use crate::alloc_addresses::EvalContextExt as _;
use crate::*;

/// Maximum size memory access in bytes that GenMC supports.
pub(super) const MAX_ACCESS_SIZE: u64 = 8;

/// Convert a [`GenmcHandlerResult`] into an `InterpResult`, raising the matching Miri error.
// FIXME(genmc): improve error handling.
pub(super) fn get_outcome<'tcx, T>(result: GenmcHandlerResult<T>) -> InterpResult<'tcx, T> {
    match result {
        // A handler producing an invalid result means that the execution is moot.
        GenmcHandlerResult::Invalid => throw_machine_stop!(TerminationInfo::GenmcMoot),
        GenmcHandlerResult::Error(e) => throw_ub_format!("{e}"),
        GenmcHandlerResult::Ok(outcome) => interp_ok(outcome),
    }
}

/// Inverse function to `scalar_to_genmc_scalar`.
///
/// Convert a Miri `Scalar` to a `GenmcScalar`.
/// To be able to restore pointer provenance from a `GenmcScalar`, the base address of the allocation of the pointer is also stored in the `GenmcScalar`.
/// We cannot use the `AllocId` instead of the base address, since Miri has no control over the `AllocId`, and it may change across executions.
/// Pointers with `Wildcard` provenance are not supported.
pub fn scalar_to_genmc_scalar<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    genmc_ctx: &GenmcCtx,
    scalar: Scalar,
) -> InterpResult<'tcx, GenmcScalar> {
    interp_ok(match scalar {
        rustc_const_eval::interpret::Scalar::Int(scalar_int) => {
            // FIXME(genmc): Add u128 support once GenMC supports it.
            let value: u64 = scalar_int.to_uint(scalar_int.size()).try_into().unwrap();
            GenmcScalar { value, provenance: 0, is_init: true }
        }
        rustc_const_eval::interpret::Scalar::Ptr(pointer, _ptr_size) => {
            // FIXME(genmc,borrow tracking): Borrow tracking information is lost.
            let addr = crate::Pointer::from(pointer).addr();
            if let crate::Provenance::Wildcard = pointer.provenance {
                throw_unsup_format!("Pointers with wildcard provenance not allowed in GenMC mode");
            }
            let (alloc_id, _size, _prov_extra) =
                rustc_const_eval::interpret::Machine::ptr_get_alloc(ecx, pointer, 0).unwrap();
            let base_addr = ecx.addr_from_alloc_id(alloc_id, None)?;
            // Add the base_addr alloc_id pair to the map.
            genmc_ctx.exec_state.genmc_shared_allocs_map.borrow_mut().insert(base_addr, alloc_id);
            GenmcScalar { value: addr.bytes(), provenance: base_addr, is_init: true }
        }
    })
}

/// Inverse function to `scalar_to_genmc_scalar`.
///
/// Convert a `GenmcScalar` back into a Miri `Scalar`.
/// For pointers, attempt to convert the stored base address of their allocation back into an `AllocId`.
pub fn genmc_scalar_to_scalar<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    genmc_ctx: &GenmcCtx,
    scalar: GenmcScalar,
    size: Size,
) -> InterpResult<'tcx, Scalar> {
    // If `provenance` is zero, we have a regular integer.
    if scalar.provenance == 0 {
        // NOTE: GenMC always returns 64 bit values, and the upper bits are not yet truncated.
        // FIXME(genmc): GenMC should be doing the truncation, not Miri.
        let (value_scalar_int, _got_truncated) = ScalarInt::truncate_from_uint(scalar.value, size);
        return interp_ok(Scalar::from(value_scalar_int));
    }
    // `provenance` is non-zero, we have a pointer.
    // When we get a pointer from GenMC, then we must have sent it to GenMC before in the same
    // execution (since the reads-from relation is always respected).
    let alloc_id = genmc_ctx.exec_state.genmc_shared_allocs_map.borrow()[&scalar.provenance];
    // FIXME(genmc,borrow tracking): Borrow tracking not yet supported.
    let provenance = machine::Provenance::Concrete { alloc_id, tag: BorTag::default() };
    let ptr = interpret::Pointer::new(provenance, Size::from_bytes(scalar.value));
    interp_ok(Scalar::from_pointer(ptr, &ecx.tcx))
}

impl AtomicReadOrd {
    pub(super) fn to_genmc(self) -> MemOrdering {
        match self {
            AtomicReadOrd::Relaxed => MemOrdering::Relaxed,
            AtomicReadOrd::Acquire => MemOrdering::Acquire,
            AtomicReadOrd::SeqCst => MemOrdering::SequentiallyConsistent,
        }
    }
}

impl AtomicWriteOrd {
    pub(super) fn to_genmc(self) -> MemOrdering {
        match self {
            AtomicWriteOrd::Relaxed => MemOrdering::Relaxed,
            AtomicWriteOrd::Release => MemOrdering::Release,
            AtomicWriteOrd::SeqCst => MemOrdering::SequentiallyConsistent,
        }
    }
}

impl AtomicFenceOrd {
    pub(super) fn to_genmc(self) -> MemOrdering {
        match self {
            AtomicFenceOrd::Acquire => MemOrdering::Acquire,
            AtomicFenceOrd::Release => MemOrdering::Release,
            AtomicFenceOrd::AcqRel => MemOrdering::AcquireRelease,
            AtomicFenceOrd::SeqCst => MemOrdering::SequentiallyConsistent,
        }
    }
}

impl AtomicRwOrd {
    pub(super) fn to_genmc(self) -> MemOrdering {
        match self {
            AtomicRwOrd::Relaxed => MemOrdering::Relaxed,
            AtomicRwOrd::Acquire => MemOrdering::Acquire,
            AtomicRwOrd::Release => MemOrdering::Release,
            AtomicRwOrd::AcqRel => MemOrdering::AcquireRelease,
            AtomicRwOrd::SeqCst => MemOrdering::SequentiallyConsistent,
        }
    }
}

/// Convert an atomic binary operation to its GenMC counterpart.
pub(super) fn to_genmc_rmw_op(atomic_op: AtomicRmwOp, is_signed: bool) -> RMWBinOp {
    match (atomic_op, is_signed) {
        (AtomicRmwOp::Min, true) => RMWBinOp::Min,
        (AtomicRmwOp::Max, true) => RMWBinOp::Max,
        (AtomicRmwOp::Min, false) => RMWBinOp::UMin,
        (AtomicRmwOp::Max, false) => RMWBinOp::UMax,
        (AtomicRmwOp::Swap, _is_signed) => RMWBinOp::Xchg,
        (AtomicRmwOp::MirOp { op, neg }, _is_signed) =>
            match (op, neg) {
                (mir::BinOp::Add, false) => RMWBinOp::Add,
                (mir::BinOp::Sub, false) => RMWBinOp::Sub,
                (mir::BinOp::BitXor, false) => RMWBinOp::Xor,
                (mir::BinOp::BitAnd, false) => RMWBinOp::And,
                (mir::BinOp::BitAnd, true) => RMWBinOp::Nand,
                (mir::BinOp::BitOr, false) => RMWBinOp::Or,
                _ => {
                    panic!("unsupported atomic operation: bin_op: {op:?}, negate: {neg}");
                }
            },
    }
}
