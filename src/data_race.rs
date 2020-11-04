//! Implementation of a data-race detector
//!  uses Lamport Timestamps / Vector-clocks
//!  base on the Dyamic Race Detection for C++:
//!     - https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf
//!  to extend data-race detection to work correctly with fences
//!  and RMW operations
//! This does not explore weak memory orders and so can still miss data-races
//!  but should not report false-positives
//! Data-race definiton from(https://en.cppreference.com/w/cpp/language/memory_model#Threads_and_data_races):
//!  - if a memory location is accessed by twice is a data-race unless:
//!    - both operations execute on the same thread/signal-handler
//!    - both conflicting operations are atomic operations (1 atomic and 1 non-atomic race)
//!    - 1 of the operations happens-before the other operation (see link for definition)

use std::{
    fmt::{self, Debug}, cmp::Ordering, rc::Rc,
    cell::{Cell, RefCell, Ref, RefMut}, ops::Index, mem
};

use rustc_index::vec::{Idx, IndexVec};
use rustc_target::abi::Size;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_data_structures::fx::FxHashMap;

use smallvec::SmallVec;

use crate::{
    MiriEvalContext, ThreadId, Tag, MiriEvalContextExt, RangeMap,
    MPlaceTy, ImmTy, InterpResult, Pointer, ScalarMaybeUninit,
    OpTy, Immediate, MemPlaceMeta
};

pub type AllocExtra = VClockAlloc;
pub type MemoryExtra = Rc<GlobalState>;

/// Valid atomic read-write operations, alias of atomic::Ordering (not non-exhaustive)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicRWOp {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Valid atomic read operations, subset of atomic::Ordering
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicReadOp {
    Relaxed,
    Acquire,
    SeqCst,
}

/// Valid atomic write operations, subset of atomic::Ordering
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicWriteOp {
    Relaxed,
    Release,
    SeqCst,
}


/// Valid atomic fence operations, subset of atomic::Ordering
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicFenceOp {
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Evaluation context extensions
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: MiriEvalContextExt<'mir, 'tcx> {

    /// Variant of `read_immediate` that does not perform `data-race` checks.
    fn read_immediate_racy(&self, op: MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx, ImmTy<'tcx, Tag>> {
        let this = self.eval_context_ref();
        let data_race = &*this.memory.extra.data_race;

        let old = data_race.multi_threaded.replace(false);
        let res = this.read_immediate(op.into());
        data_race.multi_threaded.set(old);

        res
    }
    
    /// Variant of `write_immediate` that does not perform `data-race` checks.
    fn write_immediate_racy(
        &mut self, src: Immediate<Tag>, dest: MPlaceTy<'tcx, Tag>
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let data_race = &*this.memory.extra.data_race;
        let old = data_race.multi_threaded.replace(false);

        let imm = this.write_immediate(src, dest.into());

        let data_race = &*this.memory.extra.data_race;
        data_race.multi_threaded.set(old);
        imm
    }

    /// Variant of `read_scalar` that does not perform data-race checks.
    fn read_scalar_racy(
        &self, op: MPlaceTy<'tcx, Tag>
    )-> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        Ok(self.read_immediate_racy(op)?.to_scalar_or_uninit())
    }

    /// Variant of `write_scalar` that does not perform data-race checks.
    fn write_scalar_racy(
        &mut self, val: ScalarMaybeUninit<Tag>, dest: MPlaceTy<'tcx, Tag>
    ) -> InterpResult<'tcx> {
        self.write_immediate_racy(Immediate::Scalar(val.into()), dest)
    }

    /// Variant of `read_scalar_at_offset` helper function that does not perform
    /// `data-race checks.
    fn read_scalar_at_offset_racy(
        &self,
        op: OpTy<'tcx, Tag>,
        offset: u64,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_ref();
        let op_place = this.deref_operand(op)?;
        let offset = Size::from_bytes(offset);
        // Ensure that the following read at an offset is within bounds
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, MemPlaceMeta::None, layout, this)?;
        this.read_scalar_racy(value_place.into())
    }

    /// Variant of `write_scalar_at_offfset` helper function that performs
    ///  an atomic load operation with verification instead
    fn read_scalar_at_offset_atomic(
        &mut self,
        op: OpTy<'tcx, Tag>,
        offset: u64,
        layout: TyAndLayout<'tcx>,
        atomic: AtomicReadOp
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_mut();
        let op_place = this.deref_operand(op)?;
        let offset = Size::from_bytes(offset);
        // Ensure that the following read at an offset is within bounds
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, MemPlaceMeta::None, layout, this)?;
        let res = this.read_scalar_racy(value_place.into())?;
        this.validate_atomic_load(value_place, atomic)?;
        Ok(res)
    }

    /// Variant of `write_scalar_at_offfset` helper function that does not perform
    ///  data-race checks.
    fn write_scalar_at_offset_racy(
        &mut self,
        op: OpTy<'tcx, Tag>,
        offset: u64,
        value: impl Into<ScalarMaybeUninit<Tag>>,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let op_place = this.deref_operand(op)?;
        let offset = Size::from_bytes(offset);
        // Ensure that the following read at an offset is within bounds
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, MemPlaceMeta::None, layout, this)?;
        this.write_scalar_racy(value.into(), value_place.into())
    }

    /// Load the data race allocation state for a given memory place
    ///  also returns the size and offset of the result in the allocation
    ///  metadata
    /// This is used for atomic loads since unconditionally requesteing
    ///  mutable access causes issues for read-only memory, which will
    ///  fail validation on mutable access
    fn load_data_race_state_ref<'a>(
        &'a self, place: MPlaceTy<'tcx, Tag>
    ) -> InterpResult<'tcx, (&'a VClockAlloc, Size, Size)> where 'mir: 'a {
        let this = self.eval_context_ref();

        let ptr = place.ptr.assert_ptr();
        let size = place.layout.size;
        let data_race = &this.memory.get_raw(ptr.alloc_id)?.extra.data_race;

        Ok((data_race, size, ptr.offset))
    }

    /// Load the data race allocation state for a given memory place
    ///  also returns the size and the offset of the result in the allocation
    ///  metadata
    fn load_data_race_state_mut<'a>(
        &'a mut self, place: MPlaceTy<'tcx, Tag>
    ) -> InterpResult<'tcx, (&'a mut VClockAlloc, Size, Size)> where 'mir: 'a {
        let this = self.eval_context_mut();

        let ptr = place.ptr.assert_ptr();
        let size = place.layout.size;
        let data_race = &mut this.memory.get_raw_mut(ptr.alloc_id)?.extra.data_race;

        Ok((data_race, size, ptr.offset))
    }
    
    /// Update the data-race detector for an atomic read occuring at the
    ///  associated memory-place and on the current thread
    fn validate_atomic_load(
        &mut self, place: MPlaceTy<'tcx, Tag>, atomic: AtomicReadOp
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let data_race = &*this.memory.extra.data_race;
        if data_race.multi_threaded.get() {
            data_race.advance_vector_clock();

            let (
                alloc, size, offset
            ) = this.load_data_race_state_ref(place)?;
            log::trace!(
                "Atomic load on {:?} with ordering {:?}, in memory({:?}, offset={}, size={})",
                alloc.global.current_thread(), atomic,
                place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes()
            );

            let current_thread = alloc.global.current_thread();
            let mut current_state = alloc.global.current_thread_state_mut();
            if atomic == AtomicReadOp::Relaxed {
                // Perform relaxed atomic load
                for (_,range) in alloc.alloc_ranges.borrow_mut().iter_mut(offset, size) {
                    if range.load_relaxed(&mut *current_state, current_thread) == Err(DataRace) {
                        mem::drop(current_state);
                        return VClockAlloc::report_data_race(
                            &alloc.global, range, "ATOMIC_LOAD", true,
                            place.ptr.assert_ptr(), size
                        );
                    }
                }
            }else{
                // Perform acquire(or seq-cst) atomic load
                for (_,range) in alloc.alloc_ranges.borrow_mut().iter_mut(offset, size) {
                    if range.acquire(&mut *current_state, current_thread) == Err(DataRace) {
                        mem::drop(current_state);
                        return VClockAlloc::report_data_race(
                            &alloc.global, range, "ATOMIC_LOAD", true,
                            place.ptr.assert_ptr(), size
                        );
                    }
                }
            }

            // Log changes to atomic memory
            if log::log_enabled!(log::Level::Trace) {
                for (_,range) in alloc.alloc_ranges.borrow_mut().iter(offset, size) {
                    log::trace!(
                        "  updated atomic memory({:?}, offset={}, size={}) to {:#?}",
                        place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes(),
                        range.atomic_ops
                    );
                }
            }

            mem::drop(current_state);
            let data_race = &*this.memory.extra.data_race;
            data_race.advance_vector_clock();
        }
        Ok(())
    }

    /// Update the data-race detector for an atomic write occuring at the
    ///  associated memory-place and on the current thread
    fn validate_atomic_store(
        &mut self, place: MPlaceTy<'tcx, Tag>, atomic: AtomicWriteOp
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let data_race = &*this.memory.extra.data_race;
        if data_race.multi_threaded.get() {
            data_race.advance_vector_clock();

            let (
                alloc, size, offset
            ) = this.load_data_race_state_mut(place)?;
            let current_thread = alloc.global.current_thread();
            let mut current_state = alloc.global.current_thread_state_mut();
            log::trace!(
                "Atomic store on {:?} with ordering {:?}, in memory({:?}, offset={}, size={})",
                current_thread, atomic,
                place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes()
            );

            if atomic == AtomicWriteOp::Relaxed {
                // Perform relaxed atomic store
                for (_,range) in alloc.alloc_ranges.get_mut().iter_mut(offset, size) {
                    if range.store_relaxed(&mut *current_state, current_thread) == Err(DataRace) {
                        mem::drop(current_state);
                        return VClockAlloc::report_data_race(
                            &alloc.global, range, "ATOMIC_STORE", true,
                            place.ptr.assert_ptr(), size
                        );
                    }
                }
            }else{
                // Perform release(or seq-cst) atomic store
                for (_,range) in alloc.alloc_ranges.get_mut().iter_mut(offset, size) {
                    if range.release(&mut *current_state, current_thread) == Err(DataRace) {
                        mem::drop(current_state);
                        return VClockAlloc::report_data_race(
                            &alloc.global, range, "ATOMIC_STORE", true,
                            place.ptr.assert_ptr(), size
                        );
                    }
                }
            }

            // Log changes to atomic memory
            if log::log_enabled!(log::Level::Trace) {
                for (_,range) in alloc.alloc_ranges.get_mut().iter(offset, size) {
                    log::trace!(
                        "  updated atomic memory({:?}, offset={}, size={}) to {:#?}",
                        place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes(),
                        range.atomic_ops
                    );
                }
            }

            mem::drop(current_state);
            let data_race = &*this.memory.extra.data_race;
            data_race.advance_vector_clock();
        }
        Ok(())
    }

    /// Update the data-race detector for an atomic read-modify-write occuring
    ///  at the associated memory place and on the current thread
    fn validate_atomic_rmw(
        &mut self, place: MPlaceTy<'tcx, Tag>, atomic: AtomicRWOp
    ) -> InterpResult<'tcx> {
        use AtomicRWOp::*;
        let this = self.eval_context_mut();
        let data_race = &*this.memory.extra.data_race;
        if data_race.multi_threaded.get() {
            data_race.advance_vector_clock();

            let (
                alloc, size, offset
            ) = this.load_data_race_state_mut(place)?;
            let current_thread = alloc.global.current_thread();
            let mut current_state = alloc.global.current_thread_state_mut();
            log::trace!(
                "Atomic RMW on {:?} with ordering {:?}, in memory({:?}, offset={}, size={})",
                current_thread, atomic,
                place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes()
            );

            let acquire = matches!(atomic, Acquire | AcqRel | SeqCst);
            let release = matches!(atomic, Release | AcqRel | SeqCst);
            for (_,range) in alloc.alloc_ranges.get_mut().iter_mut(offset, size) {
                //FIXME: this is probably still slightly wrong due to the quirks
                // in the c++11 memory model
                let maybe_race = if acquire {
                    // Atomic RW-Op acquire
                    range.acquire(&mut *current_state, current_thread)
                }else{
                    range.load_relaxed(&mut *current_state, current_thread) 
                };
                if maybe_race == Err(DataRace) {
                    mem::drop(current_state);
                    return VClockAlloc::report_data_race(
                        &alloc.global, range, "ATOMIC_RMW(LOAD)", true,
                        place.ptr.assert_ptr(), size
                    );
                }
                let maybe_race = if release {
                    // Atomic RW-Op release
                    range.rmw_release(&mut *current_state, current_thread)
                }else{
                    range.rmw_relaxed(&mut *current_state, current_thread)
                };
                if maybe_race == Err(DataRace) {
                    mem::drop(current_state);
                    return VClockAlloc::report_data_race(
                        &alloc.global, range, "ATOMIC_RMW(STORE)", true,
                        place.ptr.assert_ptr(), size
                    );
                }
            }

            // Log changes to atomic memory
            if log::log_enabled!(log::Level::Trace) {
                for (_,range) in alloc.alloc_ranges.get_mut().iter(offset, size) {
                    log::trace!(
                        "  updated atomic memory({:?}, offset={}, size={}) to {:#?}",
                        place.ptr.assert_ptr().alloc_id, offset.bytes(), size.bytes(),
                        range.atomic_ops
                    );
                }
            }

            mem::drop(current_state);
            let data_race = &*this.memory.extra.data_race;
            data_race.advance_vector_clock();
        }
        Ok(())
    }

    /// Update the data-race detector for an atomic fence on the current thread
    fn validate_atomic_fence(&mut self, atomic: AtomicFenceOp) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let data_race = &*this.memory.extra.data_race;
        if data_race.multi_threaded.get() {
            data_race.advance_vector_clock();

            log::trace!("Atomic fence on {:?} with ordering {:?}", data_race.current_thread(), atomic);
            // Apply data-race detection for the current fences
            //  this treats AcqRel and SeqCst as the same as a acquire
            //  and release fence applied in the same timestamp.
            if atomic != AtomicFenceOp::Release {
                // Either Acquire | AcqRel | SeqCst
                data_race.current_thread_state_mut().apply_acquire_fence();
            }
            if atomic != AtomicFenceOp::Acquire {
                // Either Release | AcqRel | SeqCst
                data_race.current_thread_state_mut().apply_release_fence();
            }

            data_race.advance_vector_clock();
        }
        Ok(())
    }
}

/// Handle for locks to express their
///  acquire-release semantics
#[derive(Clone, Debug, Default)]
pub struct DataRaceLockHandle {

    /// Internal acquire-release clock
    ///  to express the acquire release sync
    ///  found in concurrency primitives
    clock: VClock,
}
impl DataRaceLockHandle {
    pub fn set_values(&mut self, other: &Self) {
        self.clock.set_values(&other.clock)
    }
    pub fn reset(&mut self) {
        self.clock.set_zero_vector();
    }
}


/// Avoid an atomic allocation for the common
///  case with atomic operations where the number
///  of active release sequences is small
#[derive(Clone, PartialEq, Eq)]
enum AtomicReleaseSequences {

    /// Contains one or no values
    ///  if empty: (None, reset vector clock)
    ///  if one:   (Some(thread), thread_clock)
    ReleaseOneOrEmpty(Option<ThreadId>, VClock),

    /// Contains two or more values
    ///  stored in a hash-map of thread id to
    ///  vector clocks
    ReleaseMany(FxHashMap<ThreadId, VClock>)
}
impl AtomicReleaseSequences {

    /// Return an empty set of atomic release sequences
    #[inline]
    fn new() -> AtomicReleaseSequences {
        Self::ReleaseOneOrEmpty(None, VClock::default())
    }

    /// Remove all values except for the value stored at `thread` and set
    ///  the vector clock to the associated `clock` value
    #[inline]
    fn clear_and_set(&mut self, thread: ThreadId, clock: &VClock) {
        match self {
            Self::ReleaseOneOrEmpty(id, rel_clock) => {
                *id = Some(thread);
                rel_clock.set_values(clock);
            }
            Self::ReleaseMany(_) => {
                *self = Self::ReleaseOneOrEmpty(Some(thread), clock.clone());
            }
        }
    }

    /// Remove all values except for the value stored at `thread`
    #[inline]
    fn clear_and_retain(&mut self, thread: ThreadId) {
        match self {
            Self::ReleaseOneOrEmpty(id, rel_clock) => {
                // If the id is the same, then reatin the value
                //  otherwise delete and clear the release vector clock
                if *id != Some(thread) {
                    *id = None;
                    rel_clock.set_zero_vector();
                }
            },
            Self::ReleaseMany(hash_map) => {
                // Retain only the thread element, so reduce to size
                //  of 1 or 0, and move to smaller format
                if let Some(clock) = hash_map.remove(&thread) {
                    *self = Self::ReleaseOneOrEmpty(Some(thread), clock);
                }else{
                    *self = Self::new();
                }
            }
        }
    }

    /// Insert a release sequence at `thread` with values `clock`
    fn insert(&mut self, thread: ThreadId, clock: &VClock) {
        match self {
            Self::ReleaseOneOrEmpty(id, rel_clock) => {
                if id.map_or(true, |id| id == thread) {
                    *id = Some(thread);
                    rel_clock.set_values(clock);
                }else{
                    let mut hash_map = FxHashMap::default();
                    hash_map.insert(thread, clock.clone());
                    hash_map.insert(id.unwrap(), rel_clock.clone());
                    *self = Self::ReleaseMany(hash_map);
                }
            },
            Self::ReleaseMany(hash_map) => {
                hash_map.insert(thread, clock.clone());
            }
        }
    }

    /// Return the release sequence at `thread` if one exists
    #[inline]
    fn load(&self, thread: ThreadId) -> Option<&VClock> {
        match self {
            Self::ReleaseOneOrEmpty(id, clock) => {
                if *id == Some(thread) {
                    Some(clock)
                }else{
                    None
                }
            },
            Self::ReleaseMany(hash_map) => {
                hash_map.get(&thread)
            }
        }
    }
}

/// Custom debug implementation to correctly
///  print debug as a logical mapping from threads
///  to vector-clocks
impl Debug for AtomicReleaseSequences {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReleaseOneOrEmpty(None,_) => {
                f.debug_map().finish()
            },
            Self::ReleaseOneOrEmpty(Some(id), clock) => {
                f.debug_map().entry(&id, &clock).finish()
            },
            Self::ReleaseMany(hash_map) => {
                Debug::fmt(hash_map, f)
            }
        }
    }
}

/// Error returned by finding a data race
///  should be elaborated upon
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DataRace;

/// Externally stored memory cell clocks
///  explicitly to reduce memory usage for the
///  common case where no atomic operations
///  exists on the memory cell
#[derive(Clone, PartialEq, Eq, Debug)]
struct AtomicMemoryCellClocks {

    /// The clock-vector for the set of atomic read operations
    ///  used for detecting data-races with non-atomic write
    ///  operations
    read_vector: VClock,

    /// The clock-vector for the set of atomic write operations
    ///  used for detecting data-races with non-atomic read or
    ///  write operations
    write_vector: VClock,

    /// Synchronization vector for acquire-release semantics
    ///   contains the vector of timestamps that will
    ///   happen-before a thread if an acquire-load is 
    ///   performed on the data
    sync_vector: VClock,

    /// The Hash-Map of all threads for which a release
    ///  sequence exists in the memory cell, required
    ///  since read-modify-write operations do not
    ///  invalidate existing release sequences 
    release_sequences: AtomicReleaseSequences,
}

/// Memory Cell vector clock metadata
///  for data-race detection
#[derive(Clone, PartialEq, Eq, Debug)]
struct MemoryCellClocks {

    /// The vector-clock of the last write, only one value is stored
    ///  since all previous writes happened-before the current write
    write: Timestamp,

    /// The identifier of the thread that performed the last write
    ///  operation
    write_thread: ThreadId,

    /// The vector-clock of the set of previous reads
    ///  each index is set to the timestamp that the associated
    ///  thread last read this value.
    read: VClock,

    /// Atomic acquire & release sequence tracking clocks
    ///  for non-atomic memory in the common case this
    ///  value is set to None
    atomic_ops: Option<Box<AtomicMemoryCellClocks>>,
}

/// Create a default memory cell clocks instance
///  for uninitialized memory
impl Default for MemoryCellClocks {
    fn default() -> Self {
        MemoryCellClocks {
            read: VClock::default(),
            write: 0,
            write_thread: ThreadId::new(u32::MAX as usize),
            atomic_ops: None
        }
    }
}

impl MemoryCellClocks {

    /// Load the internal atomic memory cells if they exist
    #[inline]
    fn atomic(&self) -> Option<&AtomicMemoryCellClocks> {
        match &self.atomic_ops {
            Some(op) => Some(&*op),
            None => None
        }
    }

    /// Load or create the internal atomic memory metadata
    ///  if it does not exist
    #[inline]
    fn atomic_mut(&mut self) -> &mut AtomicMemoryCellClocks {
        self.atomic_ops.get_or_insert_with(|| {
            Box::new(AtomicMemoryCellClocks {
                read_vector: VClock::default(),
                write_vector: VClock::default(),
                sync_vector: VClock::default(),
                release_sequences: AtomicReleaseSequences::new()
            })
        })
    }

    /// Update memory cell data-race tracking for atomic
    ///  load acquire semantics, is a no-op if this memory was
    ///  not used previously as atomic memory
    fn acquire(&mut self, clocks: &mut ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_read_detect(clocks, thread)?;
        if let Some(atomic) = self.atomic() {
            clocks.clock.join(&atomic.sync_vector);
        }
        Ok(())
    }
    /// Update memory cell data-race tracking for atomic
    ///  load relaxed semantics, is a no-op if this memory was
    ///  not used previously as atomic memory
    fn load_relaxed(&mut self, clocks: &mut ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_read_detect(clocks, thread)?;
        if let Some(atomic) = self.atomic() {
            clocks.fence_acquire.join(&atomic.sync_vector);
        }
        Ok(())
    }


    /// Update the memory cell data-race tracking for atomic
    ///  store release semantics
    fn release(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_write_detect(clocks, thread)?;
        let atomic = self.atomic_mut();
        atomic.sync_vector.set_values(&clocks.clock);
        atomic.release_sequences.clear_and_set(thread, &clocks.clock);
        Ok(())
    }
    /// Update the memory cell data-race tracking for atomic
    ///  store relaxed semantics
    fn store_relaxed(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_write_detect(clocks, thread)?;
        let atomic = self.atomic_mut();
        atomic.sync_vector.set_values(&clocks.fence_release);
        if let Some(release) = atomic.release_sequences.load(thread) {
            atomic.sync_vector.join(release);
        }
        atomic.release_sequences.clear_and_retain(thread);
        Ok(())
    }
    /// Update the memory cell data-race tracking for atomic
    ///  store release semantics for RMW operations
    fn rmw_release(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_write_detect(clocks, thread)?;
        let atomic = self.atomic_mut();
        atomic.sync_vector.join(&clocks.clock);
        atomic.release_sequences.insert(thread, &clocks.clock);
        Ok(())
    }
    /// Update the memory cell data-race tracking for atomic
    ///  store relaxed semantics for RMW operations
    fn rmw_relaxed(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        self.atomic_write_detect(clocks, thread)?;
        let atomic = self.atomic_mut();
        atomic.sync_vector.join(&clocks.fence_release);
        Ok(())
    }
    
    /// Detect data-races with an atomic read, caused by a non-atomic write that does
    ///  not happen-before the atomic-read
    fn atomic_read_detect(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        log::trace!("Atomic read with vectors: {:#?} :: {:#?}", self, clocks);
        if self.write <= clocks.clock[self.write_thread] {
            let atomic = self.atomic_mut();
            atomic.read_vector.set_at_thread(&clocks.clock, thread);
            Ok(())
        }else{
            Err(DataRace)
        }
    }

    /// Detect data-races with an atomic write, either with a non-atomic read or with
    ///  a non-atomic write:
    fn atomic_write_detect(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        log::trace!("Atomic write with vectors: {:#?} :: {:#?}", self, clocks);
        if self.write <= clocks.clock[self.write_thread] && self.read <= clocks.clock {
            let atomic = self.atomic_mut();
            atomic.write_vector.set_at_thread(&clocks.clock, thread);
            Ok(())
        }else{
            Err(DataRace)
        }
    }

    /// Detect races for non-atomic read operations at the current memory cell
    ///  returns true if a data-race is detected
    fn read_race_detect(&mut self, clocks: &ThreadClockSet, thread: ThreadId) -> Result<(), DataRace> {
        log::trace!("Unsynchronized read with vectors: {:#?} :: {:#?}", self, clocks);
        if self.write <= clocks.clock[self.write_thread] {
            let race_free = if let Some(atomic) = self.atomic() {
                atomic.write_vector <= clocks.clock
            }else{
                true
            };
            if race_free {
                self.read.set_at_thread(&clocks.clock, thread);
                Ok(())
            }else{
                Err(DataRace)
            }
        }else{
            Err(DataRace)
        }
    }

    /// Detect races for non-atomic write operations at the current memory cell
    ///  returns true if a data-race is detected
    fn write_race_detect(&mut self, clocks: &ThreadClockSet, thread: ThreadId)  -> Result<(), DataRace> {
        log::trace!("Unsynchronized write with vectors: {:#?} :: {:#?}", self, clocks);
        if self.write <= clocks.clock[self.write_thread] && self.read <= clocks.clock {
            let race_free = if let Some(atomic) = self.atomic() {
                atomic.write_vector <= clocks.clock && atomic.read_vector <= clocks.clock
            }else{
                true
            };
            if race_free {
                self.write = clocks.clock[thread];
                self.write_thread = thread;
                self.read.set_zero_vector();
                Ok(())
            }else{
                Err(DataRace)
            }
        }else{
            Err(DataRace)
        }
    }
}

/// Vector clock metadata for a logical memory allocation
#[derive(Debug, Clone)]
pub struct VClockAlloc {

    /// Range of Vector clocks, mapping to the vector-clock
    ///  index of the last write to the bytes in this allocation
    alloc_ranges: RefCell<RangeMap<MemoryCellClocks>>,

    // Pointer to global state
    global: MemoryExtra,
}

impl VClockAlloc {

    /// Create a new data-race allocation detector
    pub fn new_allocation(global: &MemoryExtra, len: Size) -> VClockAlloc {
        VClockAlloc {
            global: Rc::clone(global),
            alloc_ranges: RefCell::new(
                RangeMap::new(len, MemoryCellClocks::default())
            )
        }
    }

    // Find an index, if one exists where the value
    //  in `l` is greater than the value in `r`
    fn find_gt_index(l: &VClock, r: &VClock) -> Option<usize> {
        let l_slice = l.as_slice();
        let r_slice = r.as_slice();
        l_slice.iter().zip(r_slice.iter())
            .enumerate()
            .find_map(|(idx, (&l, &r))| {
                if l > r { Some(idx) } else { None }
            }).or_else(|| {
                if l_slice.len() > r_slice.len() {
                    // By invariant, if l_slice is longer
                    //  then one element must be larger
                    // This just validates that this is true
                    //  and reports earlier elements first
                    let l_remainder_slice = &l_slice[r_slice.len()..];
                    let idx = l_remainder_slice.iter().enumerate()
                        .find_map(|(idx, &r)| {
                            if r == 0 { None } else { Some(idx) }
                        }).expect("Invalid VClock Invariant");
                    Some(idx)
                }else{
                    None
                }
            })
    }

    /// Report a data-race found in the program
    ///  this finds the two racing threads and the type
    ///  of data-race that occured, this will also
    ///  return info about the memory location the data-race
    ///  occured in
    #[cold]
    #[inline(never)]
    fn report_data_race<'tcx>(
        global: &MemoryExtra, range: &MemoryCellClocks,
        action: &str, is_atomic: bool,
        pointer: Pointer<Tag>, len: Size
    ) -> InterpResult<'tcx> {
        let current_thread = global.current_thread();
        let current_state = global.current_thread_state();
        let mut write_clock = VClock::default();
        let (
            other_action, other_thread, other_clock
        ) = if range.write > current_state.clock[range.write_thread] {

            // Convert the write action into the vector clock it
            //  represents for diagnostic purposes
            let wclock = write_clock.get_mut_with_min_len(
                current_state.clock.as_slice().len()
                .max(range.write_thread.to_u32() as usize + 1)
            );
            wclock[range.write_thread.to_u32() as usize] = range.write;
            ("WRITE", range.write_thread, write_clock.as_slice())
        }else if let Some(idx) = Self::find_gt_index(
            &range.read, &current_state.clock
        ){
            ("READ", ThreadId::new(idx), range.read.as_slice())
        }else if !is_atomic {
            if let Some(atomic) = range.atomic() {
                if let Some(idx) = Self::find_gt_index(
                    &atomic.write_vector, &current_state.clock
                ) {
                    ("ATOMIC_STORE", ThreadId::new(idx), atomic.write_vector.as_slice())
                }else if let Some(idx) = Self::find_gt_index(
                    &atomic.read_vector, &current_state.clock
                ) {
                    ("ATOMIC_LOAD", ThreadId::new(idx), atomic.read_vector.as_slice())
                }else{
                    unreachable!("Failed to find report data-race for non-atomic operation: no race found")
                }
            }else{
                unreachable!("Failed to report data-race for non-atomic operation: no atomic component")
            }
        }else{
            unreachable!("Failed to report data-race for atomic operation")
        };

        // Load elaborated thread information about the racing thread actions
        let current_thread_info = global.print_thread_metadata(current_thread);
        let other_thread_info = global.print_thread_metadata(other_thread);
        
        // Throw the data-race detection
        throw_ub_format!(
            "Data race detected between {} on {} and {} on {}, memory({:?},offset={},size={})\
            \n\t\t -current vector clock = {:?}\
            \n\t\t -conflicting timestamp = {:?}",
            action, current_thread_info, 
            other_action, other_thread_info,
            pointer.alloc_id, pointer.offset.bytes(), len.bytes(),
            current_state.clock,
            other_clock
        )
    }

    /// Detect data-races for an unsychronized read operation, will not perform
    ///  data-race threads if `multi-threaded` is false, either due to no threads
    ///  being created or if it is temporarily disabled during a racy read or write
    ///  operation
    pub fn read<'tcx>(&self, pointer: Pointer<Tag>, len: Size) -> InterpResult<'tcx> {
        if self.global.multi_threaded.get() {
            let current_thread = self.global.current_thread();
            let current_state = self.global.current_thread_state();

            // The alloc-ranges are not split, however changes are not going to be made
            //  to the ranges being tested, so this is ok
            let mut alloc_ranges = self.alloc_ranges.borrow_mut();
            for (_,range) in alloc_ranges.iter_mut(pointer.offset, len) {
                if range.read_race_detect(&*current_state, current_thread) == Err(DataRace) {
                    // Report data-race
                    return Self::report_data_race(
                        &self.global,range, "READ", false, pointer, len
                    );
                }
            }
            Ok(())
        }else{
            Ok(())
        }
    }
    /// Detect data-races for an unsychronized write operation, will not perform
    ///  data-race threads if `multi-threaded` is false, either due to no threads
    ///  being created or if it is temporarily disabled during a racy read or write
    ///  operation
    pub fn write<'tcx>(&mut self, pointer: Pointer<Tag>, len: Size) -> InterpResult<'tcx> {
        if self.global.multi_threaded.get() {
            let current_thread = self.global.current_thread();
            let current_state = self.global.current_thread_state();
            for (_,range) in self.alloc_ranges.get_mut().iter_mut(pointer.offset, len) {
                if range.write_race_detect(&*current_state, current_thread) == Err(DataRace) {
                    // Report data-race
                    return Self::report_data_race(
                        &self.global, range, "WRITE", false, pointer, len
                    );
                }
            }
            Ok(())
        }else{
            Ok(())
        }
    }
    /// Detect data-races for an unsychronized deallocate operation, will not perform
    ///  data-race threads if `multi-threaded` is false, either due to no threads
    ///  being created or if it is temporarily disabled during a racy read or write
    ///  operation
    pub fn deallocate<'tcx>(&mut self, pointer: Pointer<Tag>, len: Size) -> InterpResult<'tcx> {
        if self.global.multi_threaded.get() {
            let current_thread = self.global.current_thread();
            let current_state = self.global.current_thread_state();
            for (_,range) in self.alloc_ranges.get_mut().iter_mut(pointer.offset, len) {
                if range.write_race_detect(&*current_state, current_thread) == Err(DataRace) {
                    // Report data-race
                    return Self::report_data_race(
                        &self.global, range, "DEALLOCATE", false, pointer, len
                    );
                }
            }
           Ok(())
        }else{
            Ok(())
        }
    }
}

/// The current set of vector clocks describing the state
///  of a thread, contains the happens-before clock and
///  additional metadata to model atomic fence operations
#[derive(Clone, Default, Debug)]
struct ThreadClockSet {
    /// The increasing clock representing timestamps
    ///  that happen-before this thread.
    clock: VClock,

    /// The set of timestamps that will happen-before this
    ///  thread once it performs an acquire fence
    fence_acquire: VClock,

    /// The last timesamp of happens-before relations that
    ///  have been released by this thread by a fence
    fence_release: VClock,
}

impl ThreadClockSet {

    /// Apply the effects of a release fence to this
    ///  set of thread vector clocks
    #[inline]
    fn apply_release_fence(&mut self) {
        self.fence_release.set_values(&self.clock);
    }

    /// Apply the effects of a acquire fence to this
    ///  set of thread vector clocks
    #[inline]
    fn apply_acquire_fence(&mut self) {
        self.clock.join(&self.fence_acquire);
    }

    /// Increment the happens-before clock at a
    ///  known index
    #[inline]
    fn increment_clock(&mut self, thread: ThreadId) {
        self.clock.increment_thread(thread);
    }

    /// Join the happens-before clock with that of
    ///  another thread, used to model thread join
    ///  operations
    fn join_with(&mut self, other: &ThreadClockSet) {
        self.clock.join(&other.clock);
    }
}

/// Global data-race detection state, contains the currently
///  executing thread as well as the vector-clocks associated
///  with each of the threads.
#[derive(Debug, Clone)]
pub struct GlobalState {

    /// Set to true once the first additional
    ///  thread has launched, due to the dependency
    ///  between before and after a thread launch
    /// Any data-races must be recorded after this
    ///  so concurrent execution can ignore recording
    ///  any data-races
    multi_threaded: Cell<bool>,

    /// The current vector clock for all threads
    ///  this includes threads that have terminated
    ///  execution
    thread_clocks: RefCell<IndexVec<ThreadId, ThreadClockSet>>,

    /// Thread name cache for better diagnostics on the reporting
    ///  of a data-race
    thread_names: RefCell<IndexVec<ThreadId, Option<Box<str>>>>,

    /// The current thread being executed,
    ///  this is mirrored from the scheduler since
    ///  it is required for loading the current vector
    ///  clock for data-race detection
    current_thread_id: Cell<ThreadId>,
}
impl GlobalState {

    /// Create a new global state, setup with just thread-id=0
    ///  advanced to timestamp = 1
    pub fn new() -> Self {
        let mut vec = IndexVec::new();
        let thread_id = vec.push(ThreadClockSet::default());
        vec[thread_id].increment_clock(thread_id);
        GlobalState {
            multi_threaded: Cell::new(false),
            thread_clocks: RefCell::new(vec),
            thread_names: RefCell::new(IndexVec::new()),
            current_thread_id: Cell::new(thread_id),
        }
    }
    

    // Hook for thread creation, enabled multi-threaded execution and marks
    //  the current thread timestamp as happening-before the current thread
    #[inline]
    pub fn thread_created(&self, thread: ThreadId) {

        // Enable multi-threaded execution mode now that there are at least
        //  two threads
        self.multi_threaded.set(true);
        let current_thread = self.current_thread_id.get();
        let mut vectors = self.thread_clocks.borrow_mut();
        vectors.ensure_contains_elem(thread, Default::default);
        let (current, created) = vectors.pick2_mut(current_thread, thread);

        // Pre increment clocks before atomic operation
        current.increment_clock(current_thread);

        // The current thread happens-before the created thread
        //  so update the created vector clock
        created.join_with(current);

        // Post increment clocks after atomic operation
        current.increment_clock(current_thread);
        created.increment_clock(thread);
    }

    /// Hook on a thread join to update the implicit happens-before relation
    ///  between the joined thead and the current thread
    #[inline]
    pub fn thread_joined(&self, current_thread: ThreadId, join_thread: ThreadId) {
        let mut vectors = self.thread_clocks.borrow_mut();
        let (current, join) = vectors.pick2_mut(current_thread, join_thread);

        // Pre increment clocks before atomic operation
        current.increment_clock(current_thread);
        join.increment_clock(join_thread);

        // The join thread happens-before the current thread
        //   so update the current vector clock
        current.join_with(join);

        // Post increment clocks after atomic operation
        current.increment_clock(current_thread);
        join.increment_clock(join_thread);
    }

    /// Hook for updating the local tracker of the currently
    ///  enabled thread, should always be updated whenever
    ///  `active_thread` in thread.rs is updated
    #[inline]
    pub fn thread_set_active(&self, thread: ThreadId) {
        self.current_thread_id.set(thread);
    }

    /// Hook for updating the local tracker of the threads name
    ///  this should always mirror the local value in thread.rs
    ///  the thread name is used for improved diagnostics
    ///  during a data-race
    #[inline]
    pub fn thread_set_name(&self, name: String) {
        let name = name.into_boxed_str();
        let mut names = self.thread_names.borrow_mut();
        let thread = self.current_thread_id.get();
        names.ensure_contains_elem(thread, Default::default);
        names[thread] = Some(name);
    }


    /// Advance the vector clock for a thread
    ///  this is called before and after any atomic/synchronizing operations
    ///  that may manipulate state
    #[inline]
    fn advance_vector_clock(&self) {
        let thread = self.current_thread_id.get();
        let mut vectors = self.thread_clocks.borrow_mut();
        vectors[thread].increment_clock(thread);

        // Log the increment in the atomic vector clock
        log::trace!("Atomic vector clock increase for {:?} to {:?}",thread, vectors[thread].clock);
    }
    

    /// Internal utility to identify a thread stored internally
    ///  returns the id and the name for better diagnostics
    fn print_thread_metadata(&self, thread: ThreadId) -> String {
        if let Some(Some(name)) = self.thread_names.borrow().get(thread) {
            let name: &str = name;
            format!("Thread(id = {:?}, name = {:?})", thread.to_u32(), &*name)
        }else{
            format!("Thread(id = {:?})", thread.to_u32())
        }
    }


    /// Acquire a lock, express that the previous call of
    ///  `validate_lock_release` must happen before this
    pub fn validate_lock_acquire(&self, lock: &DataRaceLockHandle, thread: ThreadId) {
        let mut ref_vector = self.thread_clocks.borrow_mut();
        ref_vector[thread].increment_clock(thread);

        let clocks = &mut ref_vector[thread];
        clocks.clock.join(&lock.clock);

        ref_vector[thread].increment_clock(thread);
    }

    /// Release a lock handle, express that this happens-before
    ///  any subsequent calls to `validate_lock_acquire`
    pub fn validate_lock_release(&self, lock: &mut DataRaceLockHandle, thread: ThreadId) {
        let mut ref_vector = self.thread_clocks.borrow_mut();
        ref_vector[thread].increment_clock(thread);

        let clocks = &ref_vector[thread];
        lock.clock.set_values(&clocks.clock);

        ref_vector[thread].increment_clock(thread);
    }

    /// Release a lock handle, express that this happens-before
    ///  any subsequent calls to `validate_lock_acquire` as well
    ///  as any previous calls to this function after any
    ///  `validate_lock_release` calls
    pub fn validate_lock_release_shared(&self, lock: &mut DataRaceLockHandle, thread: ThreadId) {
        let mut ref_vector = self.thread_clocks.borrow_mut();
        ref_vector[thread].increment_clock(thread);

        let clocks = &ref_vector[thread];
        lock.clock.join(&clocks.clock);

        ref_vector[thread].increment_clock(thread);
    }

    /// Load the thread clock set associated with the current thread
    #[inline]
    fn current_thread_state(&self) -> Ref<'_, ThreadClockSet> {
        let ref_vector = self.thread_clocks.borrow();
        let thread = self.current_thread_id.get();
        Ref::map(ref_vector, |vector| &vector[thread])
    }

    /// Load the thread clock set associated with the current thread
    ///  mutably for modification
    #[inline]
    fn current_thread_state_mut(&self) -> RefMut<'_, ThreadClockSet> {
        let ref_vector = self.thread_clocks.borrow_mut();
        let thread = self.current_thread_id.get();
        RefMut::map(ref_vector, |vector| &mut vector[thread])
    }

    /// Return the current thread, should be the same
    ///  as the data-race active thread
    #[inline]
    fn current_thread(&self) -> ThreadId {
        self.current_thread_id.get()
    }
}


/// The size of the vector-clock to store inline
///  clock vectors larger than this will be stored on the heap
const SMALL_VECTOR: usize = 4;

/// The type of the time-stamps recorded in the data-race detector
///  set to a type of unsigned integer
type Timestamp = u32;

/// A vector clock for detecting data-races
///  invariants:
///   - the last element in a VClock must not be 0
///     -- this means that derive(PartialEq & Eq) is correct
///     --  as there is no implicit zero tail that might be equal
///     --  also simplifies the implementation of PartialOrd
#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct VClock(SmallVec<[Timestamp; SMALL_VECTOR]>);

impl VClock {

    /// Load the backing slice behind the clock vector.
    #[inline]
    fn as_slice(&self) -> &[Timestamp] {
        self.0.as_slice()
    }

    /// Get a mutable slice to the internal vector with minimum `min_len`
    ///  elements, to preserve invariants this vector must modify
    ///  the `min_len`-1 nth element to a non-zero value
    #[inline]
    fn get_mut_with_min_len(&mut self, min_len: usize) -> &mut [Timestamp] {
        if self.0.len() < min_len {
            self.0.resize(min_len, 0);
        }
        assert!(self.0.len() >= min_len);
        self.0.as_mut_slice()
    }

    /// Increment the vector clock at a known index
    #[inline]
    fn increment_index(&mut self, idx: usize) {
        let mut_slice = self.get_mut_with_min_len(idx + 1);
        let idx_ref = &mut mut_slice[idx];
        *idx_ref = idx_ref.checked_add(1).expect("Vector clock overflow")
    }

    // Increment the vector element representing the progress
    //  of execution in the given thread
    #[inline]
    pub fn increment_thread(&mut self, thread: ThreadId) {
        self.increment_index(thread.to_u32() as usize);
    }

    // Join the two vector-clocks together, this
    //  sets each vector-element to the maximum value
    //  of that element in either of the two source elements.
    pub fn join(&mut self, other: &Self) {
        let rhs_slice = other.as_slice();
        let lhs_slice = self.get_mut_with_min_len(rhs_slice.len());

        // Element-wise set to maximum.
        for (l, &r) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            *l = r.max(*l);
        }
    }

    /// Joins with a thread at a known index
    fn set_at_index(&mut self, other: &Self, idx: usize){
        let mut_slice = self.get_mut_with_min_len(idx + 1);
        let slice = other.as_slice();
        mut_slice[idx] = slice[idx];
    }

    /// Join with a threads vector clock only at the desired index
    ///  returns true if the value updated
    #[inline]
    pub fn set_at_thread(&mut self, other: &Self, thread: ThreadId){
        self.set_at_index(other, thread.to_u32() as usize);
    }

    /// Clear the vector to all zeros, stored as an empty internal
    ///  vector
    #[inline]
    pub fn set_zero_vector(&mut self) {
        self.0.clear();
    }

    /// Set the values stored in this vector clock
    ///  to the values stored in another.
    pub fn set_values(&mut self, new_value: &VClock) {
        let new_slice = new_value.as_slice();
        self.0.resize(new_slice.len(), 0);
        self.0.copy_from_slice(new_slice);
    }
}


impl PartialOrd for VClock {
    fn partial_cmp(&self, other: &VClock) -> Option<Ordering> {

        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // Iterate through the combined vector slice
        //  keeping track of the order that is currently possible to satisfy.
        // If an ordering relation is detected to be impossible, then bail and
        //  directly return None
        let mut iter = lhs_slice.iter().zip(rhs_slice.iter());
        let mut order = match iter.next() {
            Some((lhs, rhs)) => lhs.cmp(rhs),
            None => Ordering::Equal
        };
        for (l, r) in iter {
            match order {
                Ordering::Equal => order = l.cmp(r),
                Ordering::Less => if l > r {
                    return None
                },
                Ordering::Greater => if l < r {
                    return None
                }
            }
        }

        //Now test if either left or right have trailing elements
        // by the invariant the trailing elements have at least 1
        // non zero value, so no additional calculation is required
        // to determine the result of the PartialOrder
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        match l_len.cmp(&r_len) {
            // Equal has no additional elements: return current order
            Ordering::Equal => Some(order),
            // Right has at least 1 element > than the implicit 0,
            //  so the only valid values are Ordering::Less or None
            Ordering::Less => match order {
                Ordering::Less | Ordering::Equal => Some(Ordering::Less),
                Ordering::Greater => None
            }
            // Left has at least 1 element > than the implicit 0,
            //  so the only valid values are Ordering::Greater or None
            Ordering::Greater => match order {
                Ordering::Greater | Ordering::Equal => Some(Ordering::Greater),
                Ordering::Less => None
            }
        }
    }

    fn lt(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If l_len > r_len then at least one element
        //  in l_len is > than r_len, therefore the result
        //  is either Some(Greater) or None, so return false
        //  early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len <= r_len {
            // If any elements on the left are greater than the right
            //  then the result is None or Some(Greater), both of which
            //  return false, the earlier test asserts that no elements in the
            //  extended tail violate this assumption. Otherwise l <= r, finally
            //  the case where the values are potentially equal needs to be considered
            //  and false returned as well
            let mut equal = l_len == r_len;
            for (&l, &r) in lhs_slice.iter().zip(rhs_slice.iter()) {
                if l > r {
                    return false
                }else if l < r {
                    equal = false;
                }
            }
            !equal
        }else{
            false
        }
    }

    fn le(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If l_len > r_len then at least one element
        //  in l_len is > than r_len, therefore the result
        //  is either Some(Greater) or None, so return false
        //  early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len <= r_len {
            // If any elements on the left are greater than the right
            //  then the result is None or Some(Greater), both of which
            //  return false, the earlier test asserts that no elements in the
            //  extended tail violate this assumption. Otherwise l <= r
            !lhs_slice.iter().zip(rhs_slice.iter()).any(|(&l, &r)| l > r)
        }else{
            false
        }
    }

    fn gt(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If r_len > l_len then at least one element
        //  in r_len is > than l_len, therefore the result
        //  is either Some(Less) or None, so return false
        //  early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len >= r_len {
            // If any elements on the left are less than the right
            //  then the result is None or Some(Less), both of which
            //  return false, the earlier test asserts that no elements in the
            //  extended tail violate this assumption. Otherwise l >=, finally
            //  the case where the values are potentially equal needs to be considered
            //  and false returned as well
            let mut equal = l_len == r_len;
            for (&l, &r) in lhs_slice.iter().zip(rhs_slice.iter()) {
                if l < r {
                    return false
                }else if l > r {
                    equal = false;
                }
            }
            !equal
        }else{
            false
        }
    }

    fn ge(&self, other: &VClock) -> bool {
        // Load the values as slices
        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        // If r_len > l_len then at least one element
        //  in r_len is > than l_len, therefore the result
        //  is either Some(Less) or None, so return false
        //  early.
        let l_len = lhs_slice.len();
        let r_len = rhs_slice.len();
        if l_len >= r_len {
            // If any elements on the left are less than the right
            //  then the result is None or Some(Less), both of which
            //  return false, the earlier test asserts that no elements in the
            //  extended tail violate this assumption. Otherwise l >= r
            !lhs_slice.iter().zip(rhs_slice.iter()).any(|(&l, &r)| l < r)
        }else{
            false
        }
    }
}

impl Index<ThreadId> for VClock {
    type Output = Timestamp;

    #[inline]
    fn index(&self, index: ThreadId) -> &Timestamp {
       self.as_slice().get(index.to_u32() as usize).unwrap_or(&0)
    }
}


/// Test vector clock ordering operations
///  data-race detection is tested in the external
///  test suite
#[cfg(test)]
mod tests {
    use super::{VClock, Timestamp};
    use std::cmp::Ordering;

    #[test]
    fn test_equal() {
        let mut c1 = VClock::default();
        let mut c2 = VClock::default();
        assert_eq!(c1, c2);
        c1.increment_index(5);
        assert_ne!(c1, c2);
        c2.increment_index(53);
        assert_ne!(c1, c2);
        c1.increment_index(53);
        assert_ne!(c1, c2);
        c2.increment_index(5);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_partial_order() {
        // Small test
        assert_order(&[1], &[1], Some(Ordering::Equal));
        assert_order(&[1], &[2], Some(Ordering::Less));
        assert_order(&[2], &[1], Some(Ordering::Greater));
        assert_order(&[1], &[1,2], Some(Ordering::Less));
        assert_order(&[2], &[1,2], None);

        // Misc tests
        assert_order(&[400], &[0, 1], None);

        // Large test
        assert_order(&[0,1,2,3,4,5,6,7,8,9,10], &[0,1,2,3,4,5,6,7,8,9,10,0,0,0], Some(Ordering::Equal));
        assert_order(&[0,1,2,3,4,5,6,7,8,9,10], &[0,1,2,3,4,5,6,7,8,9,10,0,1,0], Some(Ordering::Less));
        assert_order(&[0,1,2,3,4,5,6,7,8,9,11], &[0,1,2,3,4,5,6,7,8,9,10,0,0,0], Some(Ordering::Greater));
        assert_order(&[0,1,2,3,4,5,6,7,8,9,11], &[0,1,2,3,4,5,6,7,8,9,10,0,1,0], None);
        assert_order(&[0,1,2,3,4,5,6,7,8,9,9 ], &[0,1,2,3,4,5,6,7,8,9,10,0,0,0], Some(Ordering::Less));
        assert_order(&[0,1,2,3,4,5,6,7,8,9,9 ], &[0,1,2,3,4,5,6,7,8,9,10,0,1,0], Some(Ordering::Less));
    }

    fn from_slice(mut slice: &[Timestamp]) -> VClock {
        while let Some(0) = slice.last() {
            slice = &slice[..slice.len() - 1]
        }
        VClock(smallvec::SmallVec::from_slice(slice))
    }

    fn assert_order(l: &[Timestamp], r: &[Timestamp], o: Option<Ordering>) {
        let l = from_slice(l);
        let r = from_slice(r);

        //Test partial_cmp
        let compare = l.partial_cmp(&r);
        assert_eq!(compare, o, "Invalid comparison\n l: {:?}\n r: {:?}",l,r);
        let alt_compare = r.partial_cmp(&l);
        assert_eq!(alt_compare, o.map(Ordering::reverse), "Invalid alt comparison\n l: {:?}\n r: {:?}",l,r);

        //Test operatorsm with faster implementations
        assert_eq!(
            matches!(compare,Some(Ordering::Less)), l < r,
            "Invalid (<):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(compare,Some(Ordering::Less) | Some(Ordering::Equal)), l <= r,
            "Invalid (<=):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(compare,Some(Ordering::Greater)), l > r,
            "Invalid (>):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(compare,Some(Ordering::Greater) | Some(Ordering::Equal)), l >= r,
            "Invalid (>=):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(alt_compare,Some(Ordering::Less)), r < l,
            "Invalid alt (<):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(alt_compare,Some(Ordering::Less) | Some(Ordering::Equal)), r <= l,
            "Invalid alt (<=):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(alt_compare,Some(Ordering::Greater)), r > l,
            "Invalid alt (>):\n l: {:?}\n r: {:?}",l,r
        );
        assert_eq!(
            matches!(alt_compare,Some(Ordering::Greater) | Some(Ordering::Equal)), r >= l,
            "Invalid alt (>=):\n l: {:?}\n r: {:?}",l,r
        );
    }
}
