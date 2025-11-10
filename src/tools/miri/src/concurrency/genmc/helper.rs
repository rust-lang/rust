use genmc_sys::{MemOrdering, RMWBinOp};
use rustc_abi::Size;
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::mir;
use rustc_middle::mir::interpret;
use rustc_middle::ty::ScalarInt;
use tracing::debug;

use super::GenmcScalar;
use crate::alloc_addresses::EvalContextExt as _;
use crate::intrinsics::AtomicRmwOp;
use crate::*;

/// Maximum size memory access in bytes that GenMC supports.
pub(super) const MAX_ACCESS_SIZE: u64 = 8;

/// This function is used to split up a large memory access into aligned, non-overlapping chunks of a limited size.
/// Returns an iterator over the chunks, yielding `(base address, size)` of each chunk, ordered by address.
pub fn split_access(address: Size, size: Size) -> impl Iterator<Item = (u64, u64)> {
    let start_address = address.bytes();
    let end_address = start_address + size.bytes();

    let start_address_aligned = start_address.next_multiple_of(MAX_ACCESS_SIZE);
    let end_address_aligned = (end_address / MAX_ACCESS_SIZE) * MAX_ACCESS_SIZE; // prev_multiple_of

    debug!(
        "GenMC: splitting NA memory access into {MAX_ACCESS_SIZE} byte chunks: {}B + {} * {MAX_ACCESS_SIZE}B + {}B = {size:?}",
        start_address_aligned - start_address,
        (end_address_aligned - start_address_aligned) / MAX_ACCESS_SIZE,
        end_address - end_address_aligned,
    );

    // FIXME(genmc): could make remaining accesses powers-of-2, instead of 1 byte.
    let start_chunks = (start_address..start_address_aligned).map(|address| (address, 1));
    let aligned_chunks = (start_address_aligned..end_address_aligned)
        .step_by(MAX_ACCESS_SIZE.try_into().unwrap())
        .map(|address| (address, MAX_ACCESS_SIZE));
    let end_chunks = (end_address_aligned..end_address).map(|address| (address, 1));

    start_chunks.chain(aligned_chunks).chain(end_chunks)
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
        rustc_const_eval::interpret::Scalar::Ptr(pointer, size) => {
            // FIXME(genmc,borrow tracking): Borrow tracking information is lost.
            let addr = crate::Pointer::from(pointer).addr();
            if let crate::Provenance::Wildcard = pointer.provenance {
                throw_unsup_format!("Pointers with wildcard provenance not allowed in GenMC mode");
            }
            let (alloc_id, _size, _prov_extra) =
                rustc_const_eval::interpret::Machine::ptr_get_alloc(ecx, pointer, size.into())
                    .unwrap();
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

/// Since GenMC ignores the failure memory ordering and Miri should not detect bugs that don't actually exist, we upgrade the success ordering if required.
/// This means that Miri running in GenMC mode will not explore all possible executions allowed under the RC11 memory model.
/// FIXME(genmc): remove this once GenMC properly supports the failure memory ordering.
pub(super) fn maybe_upgrade_compare_exchange_success_orderings(
    success: AtomicRwOrd,
    failure: AtomicReadOrd,
) -> AtomicRwOrd {
    use AtomicReadOrd::*;
    let (success_read, success_write) = success.split_memory_orderings();
    let upgraded_success_read = match (success_read, failure) {
        (_, SeqCst) | (SeqCst, _) => SeqCst,
        (Acquire, _) | (_, Acquire) => Acquire,
        (Relaxed, Relaxed) => Relaxed,
    };
    AtomicRwOrd::from_split_memory_orderings(upgraded_success_read, success_write)
}

impl AtomicRwOrd {
    /// Split up an atomic read-write memory ordering into a separate read and write ordering.
    pub(super) fn split_memory_orderings(self) -> (AtomicReadOrd, AtomicWriteOrd) {
        match self {
            AtomicRwOrd::Relaxed => (AtomicReadOrd::Relaxed, AtomicWriteOrd::Relaxed),
            AtomicRwOrd::Acquire => (AtomicReadOrd::Acquire, AtomicWriteOrd::Relaxed),
            AtomicRwOrd::Release => (AtomicReadOrd::Relaxed, AtomicWriteOrd::Release),
            AtomicRwOrd::AcqRel => (AtomicReadOrd::Acquire, AtomicWriteOrd::Release),
            AtomicRwOrd::SeqCst => (AtomicReadOrd::SeqCst, AtomicWriteOrd::SeqCst),
        }
    }

    /// Split up an atomic read-write memory ordering into a separate read and write ordering.
    fn from_split_memory_orderings(
        read_ordering: AtomicReadOrd,
        write_ordering: AtomicWriteOrd,
    ) -> Self {
        match (read_ordering, write_ordering) {
            (AtomicReadOrd::Relaxed, AtomicWriteOrd::Relaxed) => AtomicRwOrd::Relaxed,
            (AtomicReadOrd::Acquire, AtomicWriteOrd::Relaxed) => AtomicRwOrd::Acquire,
            (AtomicReadOrd::Relaxed, AtomicWriteOrd::Release) => AtomicRwOrd::Release,
            (AtomicReadOrd::Acquire, AtomicWriteOrd::Release) => AtomicRwOrd::AcqRel,
            (AtomicReadOrd::SeqCst, AtomicWriteOrd::SeqCst) => AtomicRwOrd::SeqCst,
            _ =>
                panic!(
                    "Unsupported memory ordering combination ({read_ordering:?}, {write_ordering:?})"
                ),
        }
    }

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
