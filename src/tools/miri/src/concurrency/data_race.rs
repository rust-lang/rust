//! Implementation of a data-race detector using Lamport Timestamps / Vector clocks
//! based on the Dynamic Race Detection for C++:
//! <https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf>
//! which does not report false-positives when fences are used, and gives better
//! accuracy in presence of read-modify-write operations.
//!
//! The implementation contains modifications to correctly model the changes to the memory model in C++20
//! regarding the weakening of release sequences: <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0982r1.html>.
//! Relaxed stores now unconditionally block all currently active release sequences and so per-thread tracking of release
//! sequences is not needed.
//!
//! The implementation also models races with memory allocation and deallocation via treating allocation and
//! deallocation as a type of write internally for detecting data-races.
//!
//! Weak memory orders are explored but not all weak behaviours are exhibited, so it can still miss data-races
//! but should not report false-positives
//!
//! Data-race definition from(<https://en.cppreference.com/w/cpp/language/memory_model#Threads_and_data_races>):
//! a data race occurs between two memory accesses if they are on different threads, at least one operation
//! is non-atomic, at least one operation is a write and neither access happens-before the other. Read the link
//! for full definition.
//!
//! This re-uses vector indexes for threads that are known to be unable to report data-races, this is valid
//! because it only re-uses vector indexes once all currently-active (not-terminated) threads have an internal
//! vector clock that happens-after the join operation of the candidate thread. Threads that have not been joined
//! on are not considered. Since the thread's vector clock will only increase and a data-race implies that
//! there is some index x where `clock[x] > thread_clock`, when this is true `clock[candidate-idx] > thread_clock`
//! can never hold and hence a data-race can never be reported in that vector index again.
//! This means that the thread-index can be safely re-used, starting on the next timestamp for the newly created
//! thread.
//!
//! The timestamps used in the data-race detector assign each sequence of non-atomic operations
//! followed by a single atomic or concurrent operation a single timestamp.
//! Write, Read, Write, ThreadJoin will be represented by a single timestamp value on a thread.
//! This is because extra increment operations between the operations in the sequence are not
//! required for accurate reporting of data-race values.
//!
//! As per the paper a threads timestamp is only incremented after a release operation is performed
//! so some atomic operations that only perform acquires do not increment the timestamp. Due to shared
//! code some atomic operations may increment the timestamp when not necessary but this has no effect
//! on the data-race detection code.

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::mem;

use rustc_abi::{Align, HasDataLayout, Size};
use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Span;

use super::vector_clock::{VClock, VTimestamp, VectorIdx};
use super::weak_memory::EvalContextExt as _;
use crate::concurrency::GlobalDataRaceHandler;
use crate::diagnostics::RacingOp;
use crate::*;

pub type AllocState = VClockAlloc;

/// Valid atomic read-write orderings, alias of atomic::Ordering (not non-exhaustive).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicRwOrd {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Valid atomic read orderings, subset of atomic::Ordering.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicReadOrd {
    Relaxed,
    Acquire,
    SeqCst,
}

/// Valid atomic write orderings, subset of atomic::Ordering.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicWriteOrd {
    Relaxed,
    Release,
    SeqCst,
}

/// Valid atomic fence orderings, subset of atomic::Ordering.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AtomicFenceOrd {
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// The current set of vector clocks describing the state
/// of a thread, contains the happens-before clock and
/// additional metadata to model atomic fence operations.
#[derive(Clone, Default, Debug)]
pub(super) struct ThreadClockSet {
    /// The increasing clock representing timestamps
    /// that happen-before this thread.
    pub(super) clock: VClock,

    /// The set of timestamps that will happen-before this
    /// thread once it performs an acquire fence.
    fence_acquire: VClock,

    /// The last timestamp of happens-before relations that
    /// have been released by this thread by a release fence.
    fence_release: VClock,

    /// Timestamps of the last SC write performed by each
    /// thread, updated when this thread performs an SC fence.
    /// This is never acquired into the thread's clock, it
    /// just limits which old writes can be seen in weak memory emulation.
    pub(super) write_seqcst: VClock,

    /// Timestamps of the last SC fence performed by each
    /// thread, updated when this thread performs an SC read.
    /// This is never acquired into the thread's clock, it
    /// just limits which old writes can be seen in weak memory emulation.
    pub(super) read_seqcst: VClock,
}

impl ThreadClockSet {
    /// Apply the effects of a release fence to this
    /// set of thread vector clocks.
    #[inline]
    fn apply_release_fence(&mut self) {
        self.fence_release.clone_from(&self.clock);
    }

    /// Apply the effects of an acquire fence to this
    /// set of thread vector clocks.
    #[inline]
    fn apply_acquire_fence(&mut self) {
        self.clock.join(&self.fence_acquire);
    }

    /// Increment the happens-before clock at a
    /// known index.
    #[inline]
    fn increment_clock(&mut self, index: VectorIdx, current_span: Span) {
        self.clock.increment_index(index, current_span);
    }

    /// Join the happens-before clock with that of
    /// another thread, used to model thread join
    /// operations.
    fn join_with(&mut self, other: &ThreadClockSet) {
        self.clock.join(&other.clock);
    }
}

/// Error returned by finding a data race
/// should be elaborated upon.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct DataRace;

/// Externally stored memory cell clocks
/// explicitly to reduce memory usage for the
/// common case where no atomic operations
/// exists on the memory cell.
#[derive(Clone, PartialEq, Eq, Debug)]
struct AtomicMemoryCellClocks {
    /// The clock-vector of the timestamp of the last atomic
    /// read operation performed by each thread.
    /// This detects potential data-races between atomic read
    /// and non-atomic write operations.
    read_vector: VClock,

    /// The clock-vector of the timestamp of the last atomic
    /// write operation performed by each thread.
    /// This detects potential data-races between atomic write
    /// and non-atomic read or write operations.
    write_vector: VClock,

    /// Synchronization vector for acquire-release semantics
    /// contains the vector of timestamps that will
    /// happen-before a thread if an acquire-load is
    /// performed on the data.
    sync_vector: VClock,

    /// The size of accesses to this atomic location.
    /// We use this to detect non-synchronized mixed-size accesses. Since all accesses must be
    /// aligned to their size, this is sufficient to detect imperfectly overlapping accesses.
    /// `None` indicates that we saw multiple different sizes, which is okay as long as all accesses are reads.
    size: Option<Size>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum AtomicAccessType {
    Load(AtomicReadOrd),
    Store,
    Rmw,
}

/// Type of a non-atomic read operation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum NaReadType {
    /// Standard unsynchronized write.
    Read,

    // An implicit read generated by a retag.
    Retag,
}

impl NaReadType {
    fn description(self) -> &'static str {
        match self {
            NaReadType::Read => "non-atomic read",
            NaReadType::Retag => "retag read",
        }
    }
}

/// Type of a non-atomic write operation: allocating memory, non-atomic writes, and
/// deallocating memory are all treated as writes for the purpose of the data-race detector.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum NaWriteType {
    /// Allocate memory.
    Allocate,

    /// Standard unsynchronized write.
    Write,

    // An implicit write generated by a retag.
    Retag,

    /// Deallocate memory.
    /// Note that when memory is deallocated first, later non-atomic accesses
    /// will be reported as use-after-free, not as data races.
    /// (Same for `Allocate` above.)
    Deallocate,
}

impl NaWriteType {
    fn description(self) -> &'static str {
        match self {
            NaWriteType::Allocate => "creating a new allocation",
            NaWriteType::Write => "non-atomic write",
            NaWriteType::Retag => "retag write",
            NaWriteType::Deallocate => "deallocation",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum AccessType {
    NaRead(NaReadType),
    NaWrite(NaWriteType),
    AtomicLoad,
    AtomicStore,
    AtomicRmw,
}

/// Per-byte vector clock metadata for data-race detection.
#[derive(Clone, PartialEq, Eq, Debug)]
struct MemoryCellClocks {
    /// The vector clock timestamp and the thread that did the last non-atomic write. We don't need
    /// a full `VClock` here, it's always a single thread and nothing synchronizes, so the effective
    /// clock is all-0 except for the thread that did the write.
    write: (VectorIdx, VTimestamp),

    /// The type of operation that the write index represents,
    /// either newly allocated memory, a non-atomic write or
    /// a deallocation of memory.
    write_type: NaWriteType,

    /// The vector clock of all non-atomic reads that happened since the last non-atomic write
    /// (i.e., we join together the "singleton" clocks corresponding to each read). It is reset to
    /// zero on each write operation.
    read: VClock,

    /// Atomic access, acquire, release sequence tracking clocks.
    /// For non-atomic memory this value is set to None.
    /// For atomic memory, each byte carries this information.
    atomic_ops: Option<Box<AtomicMemoryCellClocks>>,
}

/// Extra metadata associated with a thread.
#[derive(Debug, Clone, Default)]
struct ThreadExtraState {
    /// The current vector index in use by the
    /// thread currently, this is set to None
    /// after the vector index has been re-used
    /// and hence the value will never need to be
    /// read during data-race reporting.
    vector_index: Option<VectorIdx>,

    /// Thread termination vector clock, this
    /// is set on thread termination and is used
    /// for joining on threads since the vector_index
    /// may be re-used when the join operation occurs.
    termination_vector_clock: Option<VClock>,
}

/// Global data-race detection state, contains the currently
/// executing thread as well as the vector clocks associated
/// with each of the threads.
// FIXME: it is probably better to have one large RefCell, than to have so many small ones.
#[derive(Debug, Clone)]
pub struct GlobalState {
    /// Set to true once the first additional
    /// thread has launched, due to the dependency
    /// between before and after a thread launch.
    /// Any data-races must be recorded after this
    /// so concurrent execution can ignore recording
    /// any data-races.
    multi_threaded: Cell<bool>,

    /// A flag to mark we are currently performing
    /// a data race free action (such as atomic access)
    /// to suppress the race detector
    ongoing_action_data_race_free: Cell<bool>,

    /// Mapping of a vector index to a known set of thread
    /// clocks, this is not directly mapping from a thread id
    /// since it may refer to multiple threads.
    vector_clocks: RefCell<IndexVec<VectorIdx, ThreadClockSet>>,

    /// Mapping of a given vector index to the current thread
    /// that the execution is representing, this may change
    /// if a vector index is re-assigned to a new thread.
    vector_info: RefCell<IndexVec<VectorIdx, ThreadId>>,

    /// The mapping of a given thread to associated thread metadata.
    thread_info: RefCell<IndexVec<ThreadId, ThreadExtraState>>,

    /// Potential vector indices that could be re-used on thread creation
    /// values are inserted here on after the thread has terminated and
    /// been joined with, and hence may potentially become free
    /// for use as the index for a new thread.
    /// Elements in this set may still require the vector index to
    /// report data-races, and can only be re-used after all
    /// active vector clocks catch up with the threads timestamp.
    reuse_candidates: RefCell<FxHashSet<VectorIdx>>,

    /// We make SC fences act like RMWs on a global location.
    /// To implement that, they all release and acquire into this clock.
    last_sc_fence: RefCell<VClock>,

    /// The timestamp of last SC write performed by each thread.
    /// Threads only update their own index here!
    last_sc_write_per_thread: RefCell<VClock>,

    /// Track when an outdated (weak memory) load happens.
    pub track_outdated_loads: bool,

    /// Whether weak memory emulation is enabled
    pub weak_memory: bool,
}

impl VisitProvenance for GlobalState {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // We don't have any tags.
    }
}

impl AccessType {
    fn description(self, ty: Option<Ty<'_>>, size: Option<Size>) -> String {
        let mut msg = String::new();

        if let Some(size) = size {
            if size == Size::ZERO {
                // In this case there were multiple read accesss with different sizes and then a write.
                // We will be reporting *one* of the other reads, but we don't have enough information
                // to determine which one had which size.
                assert!(self == AccessType::AtomicLoad);
                assert!(ty.is_none());
                return format!("multiple differently-sized atomic loads, including one load");
            }
            msg.push_str(&format!("{}-byte {}", size.bytes(), msg))
        }

        msg.push_str(match self {
            AccessType::NaRead(w) => w.description(),
            AccessType::NaWrite(w) => w.description(),
            AccessType::AtomicLoad => "atomic load",
            AccessType::AtomicStore => "atomic store",
            AccessType::AtomicRmw => "atomic read-modify-write",
        });

        if let Some(ty) = ty {
            msg.push_str(&format!(" of type `{ty}`"));
        }

        msg
    }

    fn is_atomic(self) -> bool {
        match self {
            AccessType::AtomicLoad | AccessType::AtomicStore | AccessType::AtomicRmw => true,
            AccessType::NaRead(_) | AccessType::NaWrite(_) => false,
        }
    }

    fn is_read(self) -> bool {
        match self {
            AccessType::AtomicLoad | AccessType::NaRead(_) => true,
            AccessType::NaWrite(_) | AccessType::AtomicStore | AccessType::AtomicRmw => false,
        }
    }

    fn is_retag(self) -> bool {
        matches!(
            self,
            AccessType::NaRead(NaReadType::Retag) | AccessType::NaWrite(NaWriteType::Retag)
        )
    }
}

impl AtomicMemoryCellClocks {
    fn new(size: Size) -> Self {
        AtomicMemoryCellClocks {
            read_vector: Default::default(),
            write_vector: Default::default(),
            sync_vector: Default::default(),
            size: Some(size),
        }
    }
}

impl MemoryCellClocks {
    /// Create a new set of clocks representing memory allocated
    ///  at a given vector timestamp and index.
    fn new(alloc: VTimestamp, alloc_index: VectorIdx) -> Self {
        MemoryCellClocks {
            read: VClock::default(),
            write: (alloc_index, alloc),
            write_type: NaWriteType::Allocate,
            atomic_ops: None,
        }
    }

    #[inline]
    fn write_was_before(&self, other: &VClock) -> bool {
        // This is the same as `self.write() <= other` but
        // without actually manifesting a clock for `self.write`.
        self.write.1 <= other[self.write.0]
    }

    #[inline]
    fn write(&self) -> VClock {
        VClock::new_with_index(self.write.0, self.write.1)
    }

    /// Load the internal atomic memory cells if they exist.
    #[inline]
    fn atomic(&self) -> Option<&AtomicMemoryCellClocks> {
        self.atomic_ops.as_deref()
    }

    /// Load the internal atomic memory cells if they exist.
    #[inline]
    fn atomic_mut_unwrap(&mut self) -> &mut AtomicMemoryCellClocks {
        self.atomic_ops.as_deref_mut().unwrap()
    }

    /// Load or create the internal atomic memory metadata if it does not exist. Also ensures we do
    /// not do mixed-size atomic accesses, and updates the recorded atomic access size.
    fn atomic_access(
        &mut self,
        thread_clocks: &ThreadClockSet,
        size: Size,
        write: bool,
    ) -> Result<&mut AtomicMemoryCellClocks, DataRace> {
        match self.atomic_ops {
            Some(ref mut atomic) => {
                // We are good if the size is the same or all atomic accesses are before our current time.
                if atomic.size == Some(size) {
                    Ok(atomic)
                } else if atomic.read_vector <= thread_clocks.clock
                    && atomic.write_vector <= thread_clocks.clock
                {
                    // We are fully ordered after all previous accesses, so we can change the size.
                    atomic.size = Some(size);
                    Ok(atomic)
                } else if !write && atomic.write_vector <= thread_clocks.clock {
                    // This is a read, and it is ordered after the last write. It's okay for the
                    // sizes to mismatch, as long as no writes with a different size occur later.
                    atomic.size = None;
                    Ok(atomic)
                } else {
                    Err(DataRace)
                }
            }
            None => {
                self.atomic_ops = Some(Box::new(AtomicMemoryCellClocks::new(size)));
                Ok(self.atomic_ops.as_mut().unwrap())
            }
        }
    }

    /// Update memory cell data-race tracking for atomic
    /// load acquire semantics, is a no-op if this memory was
    /// not used previously as atomic memory.
    fn load_acquire(
        &mut self,
        thread_clocks: &mut ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_read_detect(thread_clocks, index, access_size)?;
        if let Some(atomic) = self.atomic() {
            thread_clocks.clock.join(&atomic.sync_vector);
        }
        Ok(())
    }

    /// Update memory cell data-race tracking for atomic
    /// load relaxed semantics, is a no-op if this memory was
    /// not used previously as atomic memory.
    fn load_relaxed(
        &mut self,
        thread_clocks: &mut ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_read_detect(thread_clocks, index, access_size)?;
        if let Some(atomic) = self.atomic() {
            thread_clocks.fence_acquire.join(&atomic.sync_vector);
        }
        Ok(())
    }

    /// Update the memory cell data-race tracking for atomic
    /// store release semantics.
    fn store_release(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_write_detect(thread_clocks, index, access_size)?;
        let atomic = self.atomic_mut_unwrap(); // initialized by `atomic_write_detect`
        atomic.sync_vector.clone_from(&thread_clocks.clock);
        Ok(())
    }

    /// Update the memory cell data-race tracking for atomic
    /// store relaxed semantics.
    fn store_relaxed(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_write_detect(thread_clocks, index, access_size)?;

        // The handling of release sequences was changed in C++20 and so
        // the code here is different to the paper since now all relaxed
        // stores block release sequences. The exception for same-thread
        // relaxed stores has been removed.
        let atomic = self.atomic_mut_unwrap();
        atomic.sync_vector.clone_from(&thread_clocks.fence_release);
        Ok(())
    }

    /// Update the memory cell data-race tracking for atomic
    /// store release semantics for RMW operations.
    fn rmw_release(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_write_detect(thread_clocks, index, access_size)?;
        let atomic = self.atomic_mut_unwrap();
        atomic.sync_vector.join(&thread_clocks.clock);
        Ok(())
    }

    /// Update the memory cell data-race tracking for atomic
    /// store relaxed semantics for RMW operations.
    fn rmw_relaxed(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        self.atomic_write_detect(thread_clocks, index, access_size)?;
        let atomic = self.atomic_mut_unwrap();
        atomic.sync_vector.join(&thread_clocks.fence_release);
        Ok(())
    }

    /// Detect data-races with an atomic read, caused by a non-atomic write that does
    /// not happen-before the atomic-read.
    fn atomic_read_detect(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        trace!("Atomic read with vectors: {:#?} :: {:#?}", self, thread_clocks);
        let atomic = self.atomic_access(thread_clocks, access_size, /*write*/ false)?;
        atomic.read_vector.set_at_index(&thread_clocks.clock, index);
        // Make sure the last non-atomic write was before this access.
        if self.write_was_before(&thread_clocks.clock) { Ok(()) } else { Err(DataRace) }
    }

    /// Detect data-races with an atomic write, either with a non-atomic read or with
    /// a non-atomic write.
    fn atomic_write_detect(
        &mut self,
        thread_clocks: &ThreadClockSet,
        index: VectorIdx,
        access_size: Size,
    ) -> Result<(), DataRace> {
        trace!("Atomic write with vectors: {:#?} :: {:#?}", self, thread_clocks);
        let atomic = self.atomic_access(thread_clocks, access_size, /*write*/ true)?;
        atomic.write_vector.set_at_index(&thread_clocks.clock, index);
        // Make sure the last non-atomic write and all non-atomic reads were before this access.
        if self.write_was_before(&thread_clocks.clock) && self.read <= thread_clocks.clock {
            Ok(())
        } else {
            Err(DataRace)
        }
    }

    /// Detect races for non-atomic read operations at the current memory cell
    /// returns true if a data-race is detected.
    fn read_race_detect(
        &mut self,
        thread_clocks: &mut ThreadClockSet,
        index: VectorIdx,
        read_type: NaReadType,
        current_span: Span,
    ) -> Result<(), DataRace> {
        trace!("Unsynchronized read with vectors: {:#?} :: {:#?}", self, thread_clocks);
        if !current_span.is_dummy() {
            thread_clocks.clock.index_mut(index).span = current_span;
        }
        thread_clocks.clock.index_mut(index).set_read_type(read_type);
        if self.write_was_before(&thread_clocks.clock) {
            // We must be ordered-after all atomic writes.
            let race_free = if let Some(atomic) = self.atomic() {
                atomic.write_vector <= thread_clocks.clock
            } else {
                true
            };
            self.read.set_at_index(&thread_clocks.clock, index);
            if race_free { Ok(()) } else { Err(DataRace) }
        } else {
            Err(DataRace)
        }
    }

    /// Detect races for non-atomic write operations at the current memory cell
    /// returns true if a data-race is detected.
    fn write_race_detect(
        &mut self,
        thread_clocks: &mut ThreadClockSet,
        index: VectorIdx,
        write_type: NaWriteType,
        current_span: Span,
    ) -> Result<(), DataRace> {
        trace!("Unsynchronized write with vectors: {:#?} :: {:#?}", self, thread_clocks);
        if !current_span.is_dummy() {
            thread_clocks.clock.index_mut(index).span = current_span;
        }
        if self.write_was_before(&thread_clocks.clock) && self.read <= thread_clocks.clock {
            let race_free = if let Some(atomic) = self.atomic() {
                atomic.write_vector <= thread_clocks.clock
                    && atomic.read_vector <= thread_clocks.clock
            } else {
                true
            };
            self.write = (index, thread_clocks.clock[index]);
            self.write_type = write_type;
            if race_free {
                self.read.set_zero_vector();
                Ok(())
            } else {
                Err(DataRace)
            }
        } else {
            Err(DataRace)
        }
    }
}

impl GlobalDataRaceHandler {
    /// Select whether data race checking is disabled. This is solely an
    /// implementation detail of `allow_data_races_*` and must not be used anywhere else!
    fn set_ongoing_action_data_race_free(&self, enable: bool) {
        match self {
            GlobalDataRaceHandler::None => {}
            GlobalDataRaceHandler::Vclocks(data_race) => {
                let old = data_race.ongoing_action_data_race_free.replace(enable);
                assert_ne!(old, enable, "cannot nest allow_data_races");
            }
            GlobalDataRaceHandler::Genmc(genmc_ctx) => {
                genmc_ctx.set_ongoing_action_data_race_free(enable);
            }
        }
    }
}

/// Evaluation context extensions.
impl<'tcx> EvalContextExt<'tcx> for MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Perform an atomic read operation at the memory location.
    fn read_scalar_atomic(
        &self,
        place: &MPlaceTy<'tcx>,
        atomic: AtomicReadOrd,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_ref();
        this.atomic_access_check(place, AtomicAccessType::Load(atomic))?;
        // This will read from the last store in the modification order of this location. In case
        // weak memory emulation is enabled, this may not be the store we will pick to actually read from and return.
        // This is fine with StackedBorrow and race checks because they don't concern metadata on
        // the *value* (including the associated provenance if this is an AtomicPtr) at this location.
        // Only metadata on the location itself is used.

        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            // FIXME(GenMC): Inform GenMC what a non-atomic read here would return, to support mixed atomics/non-atomics
            let old_val = None;
            return genmc_ctx.atomic_load(
                this,
                place.ptr().addr(),
                place.layout.size,
                atomic,
                old_val,
            );
        }

        let scalar = this.allow_data_races_ref(move |this| this.read_scalar(place))?;
        let buffered_scalar = this.buffered_atomic_read(place, atomic, scalar, || {
            this.validate_atomic_load(place, atomic)
        })?;
        interp_ok(buffered_scalar.ok_or_else(|| err_ub!(InvalidUninitBytes(None)))?)
    }

    /// Perform an atomic write operation at the memory location.
    fn write_scalar_atomic(
        &mut self,
        val: Scalar,
        dest: &MPlaceTy<'tcx>,
        atomic: AtomicWriteOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.atomic_access_check(dest, AtomicAccessType::Store)?;

        // Read the previous value so we can put it in the store buffer later.
        // The program didn't actually do a read, so suppress the memory access hooks.
        // This is also a very special exception where we just ignore an error -- if this read
        // was UB e.g. because the memory is uninitialized, we don't want to know!
        let old_val = this.run_for_validation_mut(|this| this.read_scalar(dest)).discard_err();
        // Inform GenMC about the atomic store.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            // FIXME(GenMC): Inform GenMC what a non-atomic read here would return, to support mixed atomics/non-atomics
            genmc_ctx.atomic_store(this, dest.ptr().addr(), dest.layout.size, val, atomic)?;
            return interp_ok(());
        }
        this.allow_data_races_mut(move |this| this.write_scalar(val, dest))?;
        this.validate_atomic_store(dest, atomic)?;
        this.buffered_atomic_write(val, dest, atomic, old_val)
    }

    /// Perform an atomic RMW operation on a memory location.
    fn atomic_rmw_op_immediate(
        &mut self,
        place: &MPlaceTy<'tcx>,
        rhs: &ImmTy<'tcx>,
        op: mir::BinOp,
        not: bool,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        let this = self.eval_context_mut();
        this.atomic_access_check(place, AtomicAccessType::Rmw)?;

        let old = this.allow_data_races_mut(|this| this.read_immediate(place))?;

        // Inform GenMC about the atomic rmw operation.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            // FIXME(GenMC): Inform GenMC what a non-atomic read here would return, to support mixed atomics/non-atomics
            let (old_val, new_val) = genmc_ctx.atomic_rmw_op(
                this,
                place.ptr().addr(),
                place.layout.size,
                atomic,
                (op, not),
                rhs.to_scalar(),
            )?;
            this.allow_data_races_mut(|this| this.write_scalar(new_val, place))?;
            return interp_ok(ImmTy::from_scalar(old_val, old.layout));
        }

        let val = this.binary_op(op, &old, rhs)?;
        let val = if not { this.unary_op(mir::UnOp::Not, &val)? } else { val };
        this.allow_data_races_mut(|this| this.write_immediate(*val, place))?;

        this.validate_atomic_rmw(place, atomic)?;

        this.buffered_atomic_rmw(val.to_scalar(), place, atomic, old.to_scalar())?;
        interp_ok(old)
    }

    /// Perform an atomic exchange with a memory place and a new
    /// scalar value, the old value is returned.
    fn atomic_exchange_scalar(
        &mut self,
        place: &MPlaceTy<'tcx>,
        new: Scalar,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.atomic_access_check(place, AtomicAccessType::Rmw)?;

        let old = this.allow_data_races_mut(|this| this.read_scalar(place))?;
        this.allow_data_races_mut(|this| this.write_scalar(new, place))?;

        // Inform GenMC about the atomic atomic exchange.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            // FIXME(GenMC): Inform GenMC what a non-atomic read here would return, to support mixed atomics/non-atomics
            let (old_val, _is_success) = genmc_ctx.atomic_exchange(
                this,
                place.ptr().addr(),
                place.layout.size,
                new,
                atomic,
            )?;
            return interp_ok(old_val);
        }

        this.validate_atomic_rmw(place, atomic)?;

        this.buffered_atomic_rmw(new, place, atomic, old)?;
        interp_ok(old)
    }

    /// Perform an conditional atomic exchange with a memory place and a new
    /// scalar value, the old value is returned.
    fn atomic_min_max_scalar(
        &mut self,
        place: &MPlaceTy<'tcx>,
        rhs: ImmTy<'tcx>,
        min: bool,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        let this = self.eval_context_mut();
        this.atomic_access_check(place, AtomicAccessType::Rmw)?;

        let old = this.allow_data_races_mut(|this| this.read_immediate(place))?;

        // Inform GenMC about the atomic min/max operation.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            // FIXME(GenMC): Inform GenMC what a non-atomic read here would return, to support mixed atomics/non-atomics
            let (old_val, new_val) = genmc_ctx.atomic_min_max_op(
                this,
                place.ptr().addr(),
                place.layout.size,
                atomic,
                min,
                old.layout.backend_repr.is_signed(),
                rhs.to_scalar(),
            )?;
            this.allow_data_races_mut(|this| this.write_scalar(new_val, place))?;
            return interp_ok(ImmTy::from_scalar(old_val, old.layout));
        }

        let lt = this.binary_op(mir::BinOp::Lt, &old, &rhs)?.to_scalar().to_bool()?;

        #[rustfmt::skip] // rustfmt makes this unreadable
        let new_val = if min {
            if lt { &old } else { &rhs }
        } else {
            if lt { &rhs } else { &old }
        };

        this.allow_data_races_mut(|this| this.write_immediate(**new_val, place))?;

        this.validate_atomic_rmw(place, atomic)?;

        this.buffered_atomic_rmw(new_val.to_scalar(), place, atomic, old.to_scalar())?;

        // Return the old value.
        interp_ok(old)
    }

    /// Perform an atomic compare and exchange at a given memory location.
    /// On success an atomic RMW operation is performed and on failure
    /// only an atomic read occurs. If `can_fail_spuriously` is true,
    /// then we treat it as a "compare_exchange_weak" operation, and
    /// some portion of the time fail even when the values are actually
    /// identical.
    fn atomic_compare_exchange_scalar(
        &mut self,
        place: &MPlaceTy<'tcx>,
        expect_old: &ImmTy<'tcx>,
        new: Scalar,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx, Immediate<Provenance>> {
        use rand::Rng as _;
        let this = self.eval_context_mut();
        this.atomic_access_check(place, AtomicAccessType::Rmw)?;

        // Failure ordering cannot be stronger than success ordering, therefore first attempt
        // to read with the failure ordering and if successful then try again with the success
        // read ordering and write in the success case.
        // Read as immediate for the sake of `binary_op()`
        let old = this.allow_data_races_mut(|this| this.read_immediate(place))?;

        // Inform GenMC about the atomic atomic compare exchange.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            let (old, cmpxchg_success) = genmc_ctx.atomic_compare_exchange(
                this,
                place.ptr().addr(),
                place.layout.size,
                this.read_scalar(expect_old)?,
                new,
                success,
                fail,
                can_fail_spuriously,
            )?;
            if cmpxchg_success {
                this.allow_data_races_mut(|this| this.write_scalar(new, place))?;
            }
            return interp_ok(Immediate::ScalarPair(old, Scalar::from_bool(cmpxchg_success)));
        }

        // `binary_op` will bail if either of them is not a scalar.
        let eq = this.binary_op(mir::BinOp::Eq, &old, expect_old)?;
        // If the operation would succeed, but is "weak", fail some portion
        // of the time, based on `success_rate`.
        let success_rate = 1.0 - this.machine.cmpxchg_weak_failure_rate;
        let cmpxchg_success = eq.to_scalar().to_bool()?
            && if can_fail_spuriously {
                this.machine.rng.get_mut().random_bool(success_rate)
            } else {
                true
            };
        let res = Immediate::ScalarPair(old.to_scalar(), Scalar::from_bool(cmpxchg_success));

        // Update ptr depending on comparison.
        // if successful, perform a full rw-atomic validation
        // otherwise treat this as an atomic load with the fail ordering.
        if cmpxchg_success {
            this.allow_data_races_mut(|this| this.write_scalar(new, place))?;
            this.validate_atomic_rmw(place, success)?;
            this.buffered_atomic_rmw(new, place, success, old.to_scalar())?;
        } else {
            this.validate_atomic_load(place, fail)?;
            // A failed compare exchange is equivalent to a load, reading from the latest store
            // in the modification order.
            // Since `old` is only a value and not the store element, we need to separately
            // find it in our store buffer and perform load_impl on it.
            this.perform_read_on_buffered_latest(place, fail)?;
        }

        // Return the old value.
        interp_ok(res)
    }

    /// Update the data-race detector for an atomic fence on the current thread.
    fn atomic_fence(&mut self, atomic: AtomicFenceOrd) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let machine = &this.machine;
        match &this.machine.data_race {
            GlobalDataRaceHandler::None => interp_ok(()),
            GlobalDataRaceHandler::Vclocks(data_race) => data_race.atomic_fence(machine, atomic),
            GlobalDataRaceHandler::Genmc(genmc_ctx) => genmc_ctx.atomic_fence(machine, atomic),
        }
    }

    /// Calls the callback with the "release" clock of the current thread.
    /// Other threads can acquire this clock in the future to establish synchronization
    /// with this program point.
    ///
    /// The closure will only be invoked if data race handling is on.
    fn release_clock<R>(&self, callback: impl FnOnce(&VClock) -> R) -> Option<R> {
        let this = self.eval_context_ref();
        Some(
            this.machine.data_race.as_vclocks_ref()?.release_clock(&this.machine.threads, callback),
        )
    }

    /// Acquire the given clock into the current thread, establishing synchronization with
    /// the moment when that clock snapshot was taken via `release_clock`.
    fn acquire_clock(&self, clock: &VClock) {
        let this = self.eval_context_ref();
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.acquire_clock(clock, &this.machine.threads);
        }
    }
}

/// Vector clock metadata for a logical memory allocation.
#[derive(Debug, Clone)]
pub struct VClockAlloc {
    /// Assigning each byte a MemoryCellClocks.
    alloc_ranges: RefCell<DedupRangeMap<MemoryCellClocks>>,
}

impl VisitProvenance for VClockAlloc {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // No tags or allocIds here.
    }
}

impl VClockAlloc {
    /// Create a new data-race detector for newly allocated memory.
    pub fn new_allocation(
        global: &GlobalState,
        thread_mgr: &ThreadManager<'_>,
        len: Size,
        kind: MemoryKind,
        current_span: Span,
    ) -> VClockAlloc {
        // Determine the thread that did the allocation, and when it did it.
        let (alloc_timestamp, alloc_index) = match kind {
            // User allocated and stack memory should track allocation.
            MemoryKind::Machine(
                MiriMemoryKind::Rust
                | MiriMemoryKind::Miri
                | MiriMemoryKind::C
                | MiriMemoryKind::WinHeap
                | MiriMemoryKind::WinLocal
                | MiriMemoryKind::Mmap,
            )
            | MemoryKind::Stack => {
                let (alloc_index, clocks) = global.active_thread_state(thread_mgr);
                let mut alloc_timestamp = clocks.clock[alloc_index];
                alloc_timestamp.span = current_span;
                (alloc_timestamp, alloc_index)
            }
            // Other global memory should trace races but be allocated at the 0 timestamp
            // (conceptually they are allocated on the main thread before everything).
            MemoryKind::Machine(
                MiriMemoryKind::Global
                | MiriMemoryKind::Machine
                | MiriMemoryKind::Runtime
                | MiriMemoryKind::ExternStatic
                | MiriMemoryKind::Tls,
            )
            | MemoryKind::CallerLocation =>
                (VTimestamp::ZERO, global.thread_index(ThreadId::MAIN_THREAD)),
        };
        VClockAlloc {
            alloc_ranges: RefCell::new(DedupRangeMap::new(
                len,
                MemoryCellClocks::new(alloc_timestamp, alloc_index),
            )),
        }
    }

    // Find an index, if one exists where the value
    // in `l` is greater than the value in `r`.
    fn find_gt_index(l: &VClock, r: &VClock) -> Option<VectorIdx> {
        trace!("Find index where not {:?} <= {:?}", l, r);
        let l_slice = l.as_slice();
        let r_slice = r.as_slice();
        l_slice
            .iter()
            .zip(r_slice.iter())
            .enumerate()
            .find_map(|(idx, (&l, &r))| if l > r { Some(idx) } else { None })
            .or_else(|| {
                if l_slice.len() > r_slice.len() {
                    // By invariant, if l_slice is longer
                    // then one element must be larger.
                    // This just validates that this is true
                    // and reports earlier elements first.
                    let l_remainder_slice = &l_slice[r_slice.len()..];
                    let idx = l_remainder_slice
                        .iter()
                        .enumerate()
                        .find_map(|(idx, &r)| if r == VTimestamp::ZERO { None } else { Some(idx) })
                        .expect("Invalid VClock Invariant");
                    Some(idx + r_slice.len())
                } else {
                    None
                }
            })
            .map(VectorIdx::new)
    }

    /// Report a data-race found in the program.
    /// This finds the two racing threads and the type
    /// of data-race that occurred. This will also
    /// return info about the memory location the data-race
    /// occurred in. The `ty` parameter is used for diagnostics, letting
    /// the user know which type was involved in the access.
    #[cold]
    #[inline(never)]
    fn report_data_race<'tcx>(
        global: &GlobalState,
        thread_mgr: &ThreadManager<'_>,
        mem_clocks: &MemoryCellClocks,
        access: AccessType,
        access_size: Size,
        ptr_dbg: interpret::Pointer<AllocId>,
        ty: Option<Ty<'_>>,
    ) -> InterpResult<'tcx> {
        let (active_index, active_clocks) = global.active_thread_state(thread_mgr);
        let mut other_size = None; // if `Some`, this was a size-mismatch race
        let write_clock;
        let (other_access, other_thread, other_clock) =
            // First check the atomic-nonatomic cases.
            if !access.is_atomic() &&
                let Some(atomic) = mem_clocks.atomic() &&
                let Some(idx) = Self::find_gt_index(&atomic.write_vector, &active_clocks.clock)
            {
                (AccessType::AtomicStore, idx, &atomic.write_vector)
            } else if !access.is_atomic() &&
                let Some(atomic) = mem_clocks.atomic() &&
                let Some(idx) = Self::find_gt_index(&atomic.read_vector, &active_clocks.clock)
            {
                (AccessType::AtomicLoad, idx, &atomic.read_vector)
            // Then check races with non-atomic writes/reads.
            } else if mem_clocks.write.1 > active_clocks.clock[mem_clocks.write.0] {
                write_clock = mem_clocks.write();
                (AccessType::NaWrite(mem_clocks.write_type), mem_clocks.write.0, &write_clock)
            } else if let Some(idx) = Self::find_gt_index(&mem_clocks.read, &active_clocks.clock) {
                (AccessType::NaRead(mem_clocks.read[idx].read_type()), idx, &mem_clocks.read)
            // Finally, mixed-size races.
            } else if access.is_atomic() && let Some(atomic) = mem_clocks.atomic() && atomic.size != Some(access_size) {
                // This is only a race if we are not synchronized with all atomic accesses, so find
                // the one we are not synchronized with.
                other_size = Some(atomic.size.unwrap_or(Size::ZERO));
                if let Some(idx) = Self::find_gt_index(&atomic.write_vector, &active_clocks.clock)
                    {
                        (AccessType::AtomicStore, idx, &atomic.write_vector)
                    } else if let Some(idx) =
                        Self::find_gt_index(&atomic.read_vector, &active_clocks.clock)
                    {
                        (AccessType::AtomicLoad, idx, &atomic.read_vector)
                    } else {
                        unreachable!(
                            "Failed to report data-race for mixed-size access: no race found"
                        )
                    }
            } else {
                unreachable!("Failed to report data-race")
            };

        // Load elaborated thread information about the racing thread actions.
        let active_thread_info = global.print_thread_metadata(thread_mgr, active_index);
        let other_thread_info = global.print_thread_metadata(thread_mgr, other_thread);
        let involves_non_atomic = !access.is_atomic() || !other_access.is_atomic();

        // Throw the data-race detection.
        let extra = if other_size.is_some() {
            assert!(!involves_non_atomic);
            Some("overlapping unsynchronized atomic accesses must use the same access size")
        } else if access.is_read() && other_access.is_read() {
            panic!("there should be no same-size read-read races")
        } else {
            None
        };
        Err(err_machine_stop!(TerminationInfo::DataRace {
            involves_non_atomic,
            extra,
            retag_explain: access.is_retag() || other_access.is_retag(),
            ptr: ptr_dbg,
            op1: RacingOp {
                action: other_access.description(None, other_size),
                thread_info: other_thread_info,
                span: other_clock.as_slice()[other_thread.index()].span_data(),
            },
            op2: RacingOp {
                action: access.description(ty, other_size.map(|_| access_size)),
                thread_info: active_thread_info,
                span: active_clocks.clock.as_slice()[active_index.index()].span_data(),
            },
        }))?
    }

    /// Detect data-races for an unsynchronized read operation. It will not perform
    /// data-race detection if `race_detecting()` is false, either due to no threads
    /// being created or if it is temporarily disabled during a racy read or write
    /// operation for which data-race detection is handled separately, for example
    /// atomic read operations. The `ty` parameter is used for diagnostics, letting
    /// the user know which type was read.
    pub fn read<'tcx>(
        &self,
        alloc_id: AllocId,
        access_range: AllocRange,
        read_type: NaReadType,
        ty: Option<Ty<'_>>,
        machine: &MiriMachine<'_>,
    ) -> InterpResult<'tcx> {
        let current_span = machine.current_span();
        let global = machine.data_race.as_vclocks_ref().unwrap();
        if !global.race_detecting() {
            return interp_ok(());
        }
        let (index, mut thread_clocks) = global.active_thread_state_mut(&machine.threads);
        let mut alloc_ranges = self.alloc_ranges.borrow_mut();
        for (mem_clocks_range, mem_clocks) in
            alloc_ranges.iter_mut(access_range.start, access_range.size)
        {
            if let Err(DataRace) =
                mem_clocks.read_race_detect(&mut thread_clocks, index, read_type, current_span)
            {
                drop(thread_clocks);
                // Report data-race.
                return Self::report_data_race(
                    global,
                    &machine.threads,
                    mem_clocks,
                    AccessType::NaRead(read_type),
                    access_range.size,
                    interpret::Pointer::new(alloc_id, Size::from_bytes(mem_clocks_range.start)),
                    ty,
                );
            }
        }
        interp_ok(())
    }

    /// Detect data-races for an unsynchronized write operation. It will not perform
    /// data-race detection if `race_detecting()` is false, either due to no threads
    /// being created or if it is temporarily disabled during a racy read or write
    /// operation. The `ty` parameter is used for diagnostics, letting
    /// the user know which type was written.
    pub fn write<'tcx>(
        &mut self,
        alloc_id: AllocId,
        access_range: AllocRange,
        write_type: NaWriteType,
        ty: Option<Ty<'_>>,
        machine: &mut MiriMachine<'_>,
    ) -> InterpResult<'tcx> {
        let current_span = machine.current_span();
        let global = machine.data_race.as_vclocks_mut().unwrap();
        if !global.race_detecting() {
            return interp_ok(());
        }
        let (index, mut thread_clocks) = global.active_thread_state_mut(&machine.threads);
        for (mem_clocks_range, mem_clocks) in
            self.alloc_ranges.get_mut().iter_mut(access_range.start, access_range.size)
        {
            if let Err(DataRace) =
                mem_clocks.write_race_detect(&mut thread_clocks, index, write_type, current_span)
            {
                drop(thread_clocks);
                // Report data-race
                return Self::report_data_race(
                    global,
                    &machine.threads,
                    mem_clocks,
                    AccessType::NaWrite(write_type),
                    access_range.size,
                    interpret::Pointer::new(alloc_id, Size::from_bytes(mem_clocks_range.start)),
                    ty,
                );
            }
        }
        interp_ok(())
    }
}

/// Vector clock state for a stack frame (tracking the local variables
/// that do not have an allocation yet).
#[derive(Debug, Default)]
pub struct FrameState {
    local_clocks: RefCell<FxHashMap<mir::Local, LocalClocks>>,
}

/// Stripped-down version of [`MemoryCellClocks`] for the clocks we need to keep track
/// of in a local that does not yet have addressable memory -- and hence can only
/// be accessed from the thread its stack frame belongs to, and cannot be access atomically.
#[derive(Debug)]
struct LocalClocks {
    write: VTimestamp,
    write_type: NaWriteType,
    read: VTimestamp,
}

impl Default for LocalClocks {
    fn default() -> Self {
        Self { write: VTimestamp::ZERO, write_type: NaWriteType::Allocate, read: VTimestamp::ZERO }
    }
}

impl FrameState {
    pub fn local_write(&self, local: mir::Local, storage_live: bool, machine: &MiriMachine<'_>) {
        let current_span = machine.current_span();
        let global = machine.data_race.as_vclocks_ref().unwrap();
        if !global.race_detecting() {
            return;
        }
        let (index, mut thread_clocks) = global.active_thread_state_mut(&machine.threads);
        // This should do the same things as `MemoryCellClocks::write_race_detect`.
        if !current_span.is_dummy() {
            thread_clocks.clock.index_mut(index).span = current_span;
        }
        let mut clocks = self.local_clocks.borrow_mut();
        if storage_live {
            let new_clocks = LocalClocks {
                write: thread_clocks.clock[index],
                write_type: NaWriteType::Allocate,
                read: VTimestamp::ZERO,
            };
            // There might already be an entry in the map for this, if the local was previously
            // live already.
            clocks.insert(local, new_clocks);
        } else {
            // This can fail to exist if `race_detecting` was false when the allocation
            // occurred, in which case we can backdate this to the beginning of time.
            let clocks = clocks.entry(local).or_default();
            clocks.write = thread_clocks.clock[index];
            clocks.write_type = NaWriteType::Write;
        }
    }

    pub fn local_read(&self, local: mir::Local, machine: &MiriMachine<'_>) {
        let current_span = machine.current_span();
        let global = machine.data_race.as_vclocks_ref().unwrap();
        if !global.race_detecting() {
            return;
        }
        let (index, mut thread_clocks) = global.active_thread_state_mut(&machine.threads);
        // This should do the same things as `MemoryCellClocks::read_race_detect`.
        if !current_span.is_dummy() {
            thread_clocks.clock.index_mut(index).span = current_span;
        }
        thread_clocks.clock.index_mut(index).set_read_type(NaReadType::Read);
        // This can fail to exist if `race_detecting` was false when the allocation
        // occurred, in which case we can backdate this to the beginning of time.
        let mut clocks = self.local_clocks.borrow_mut();
        let clocks = clocks.entry(local).or_default();
        clocks.read = thread_clocks.clock[index];
    }

    pub fn local_moved_to_memory(
        &self,
        local: mir::Local,
        alloc: &mut VClockAlloc,
        machine: &MiriMachine<'_>,
    ) {
        let global = machine.data_race.as_vclocks_ref().unwrap();
        if !global.race_detecting() {
            return;
        }
        let (index, _thread_clocks) = global.active_thread_state_mut(&machine.threads);
        // Get the time the last write actually happened. This can fail to exist if
        // `race_detecting` was false when the write occurred, in that case we can backdate this
        // to the beginning of time.
        let local_clocks = self.local_clocks.borrow_mut().remove(&local).unwrap_or_default();
        for (_mem_clocks_range, mem_clocks) in alloc.alloc_ranges.get_mut().iter_mut_all() {
            // The initialization write for this already happened, just at the wrong timestamp.
            // Check that the thread index matches what we expect.
            assert_eq!(mem_clocks.write.0, index);
            // Convert the local's clocks into memory clocks.
            mem_clocks.write = (index, local_clocks.write);
            mem_clocks.write_type = local_clocks.write_type;
            mem_clocks.read = VClock::new_with_index(index, local_clocks.read);
        }
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Temporarily allow data-races to occur. This should only be used in
    /// one of these cases:
    /// - One of the appropriate `validate_atomic` functions will be called to
    ///   treat a memory access as atomic.
    /// - The memory being accessed should be treated as internal state, that
    ///   cannot be accessed by the interpreted program.
    /// - Execution of the interpreted program execution has halted.
    #[inline]
    fn allow_data_races_ref<R>(&self, op: impl FnOnce(&MiriInterpCx<'tcx>) -> R) -> R {
        let this = self.eval_context_ref();
        this.machine.data_race.set_ongoing_action_data_race_free(true);
        let result = op(this);
        this.machine.data_race.set_ongoing_action_data_race_free(false);
        result
    }

    /// Same as `allow_data_races_ref`, this temporarily disables any data-race detection and
    /// so should only be used for atomic operations or internal state that the program cannot
    /// access.
    #[inline]
    fn allow_data_races_mut<R>(&mut self, op: impl FnOnce(&mut MiriInterpCx<'tcx>) -> R) -> R {
        let this = self.eval_context_mut();
        this.machine.data_race.set_ongoing_action_data_race_free(true);
        let result = op(this);
        this.machine.data_race.set_ongoing_action_data_race_free(false);
        result
    }

    /// Checks that an atomic access is legal at the given place.
    fn atomic_access_check(
        &self,
        place: &MPlaceTy<'tcx>,
        access_type: AtomicAccessType,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_align(place.ptr(), align)?;
        // Ensure the allocation is mutable. Even failing (read-only) compare_exchange need mutable
        // memory on many targets (i.e., they segfault if that memory is mapped read-only), and
        // atomic loads can be implemented via compare_exchange on some targets. There could
        // possibly be some very specific exceptions to this, see
        // <https://github.com/rust-lang/miri/pull/2464#discussion_r939636130> for details.
        // We avoid `get_ptr_alloc` since we do *not* want to run the access hooks -- the actual
        // access will happen later.
        let (alloc_id, _offset, _prov) = this
            .ptr_try_get_alloc_id(place.ptr(), 0)
            .expect("there are no zero-sized atomic accesses");
        if this.get_alloc_mutability(alloc_id)? == Mutability::Not {
            // See if this is fine.
            match access_type {
                AtomicAccessType::Rmw | AtomicAccessType::Store => {
                    throw_ub_format!(
                        "atomic store and read-modify-write operations cannot be performed on read-only memory\n\
                        see <https://doc.rust-lang.org/nightly/std/sync/atomic/index.html#atomic-accesses-to-read-only-memory> for more information"
                    );
                }
                AtomicAccessType::Load(_)
                    if place.layout.size > this.tcx.data_layout().pointer_size() =>
                {
                    throw_ub_format!(
                        "large atomic load operations cannot be performed on read-only memory\n\
                        these operations often have to be implemented using read-modify-write operations, which require writeable memory\n\
                        see <https://doc.rust-lang.org/nightly/std/sync/atomic/index.html#atomic-accesses-to-read-only-memory> for more information"
                    );
                }
                AtomicAccessType::Load(o) if o != AtomicReadOrd::Relaxed => {
                    throw_ub_format!(
                        "non-relaxed atomic load operations cannot be performed on read-only memory\n\
                        these operations sometimes have to be implemented using read-modify-write operations, which require writeable memory\n\
                        see <https://doc.rust-lang.org/nightly/std/sync/atomic/index.html#atomic-accesses-to-read-only-memory> for more information"
                    );
                }
                _ => {
                    // Large relaxed loads are fine!
                }
            }
        }
        interp_ok(())
    }

    /// Update the data-race detector for an atomic read occurring at the
    /// associated memory-place and on the current thread.
    fn validate_atomic_load(
        &self,
        place: &MPlaceTy<'tcx>,
        atomic: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        this.validate_atomic_op(
            place,
            atomic,
            AccessType::AtomicLoad,
            move |memory, clocks, index, atomic| {
                if atomic == AtomicReadOrd::Relaxed {
                    memory.load_relaxed(&mut *clocks, index, place.layout.size)
                } else {
                    memory.load_acquire(&mut *clocks, index, place.layout.size)
                }
            },
        )
    }

    /// Update the data-race detector for an atomic write occurring at the
    /// associated memory-place and on the current thread.
    fn validate_atomic_store(
        &mut self,
        place: &MPlaceTy<'tcx>,
        atomic: AtomicWriteOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.validate_atomic_op(
            place,
            atomic,
            AccessType::AtomicStore,
            move |memory, clocks, index, atomic| {
                if atomic == AtomicWriteOrd::Relaxed {
                    memory.store_relaxed(clocks, index, place.layout.size)
                } else {
                    memory.store_release(clocks, index, place.layout.size)
                }
            },
        )
    }

    /// Update the data-race detector for an atomic read-modify-write occurring
    /// at the associated memory place and on the current thread.
    fn validate_atomic_rmw(
        &mut self,
        place: &MPlaceTy<'tcx>,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        use AtomicRwOrd::*;
        let acquire = matches!(atomic, Acquire | AcqRel | SeqCst);
        let release = matches!(atomic, Release | AcqRel | SeqCst);
        let this = self.eval_context_mut();
        this.validate_atomic_op(
            place,
            atomic,
            AccessType::AtomicRmw,
            move |memory, clocks, index, _| {
                if acquire {
                    memory.load_acquire(clocks, index, place.layout.size)?;
                } else {
                    memory.load_relaxed(clocks, index, place.layout.size)?;
                }
                if release {
                    memory.rmw_release(clocks, index, place.layout.size)
                } else {
                    memory.rmw_relaxed(clocks, index, place.layout.size)
                }
            },
        )
    }

    /// Generic atomic operation implementation
    fn validate_atomic_op<A: Debug + Copy>(
        &self,
        place: &MPlaceTy<'tcx>,
        atomic: A,
        access: AccessType,
        mut op: impl FnMut(
            &mut MemoryCellClocks,
            &mut ThreadClockSet,
            VectorIdx,
            A,
        ) -> Result<(), DataRace>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        assert!(access.is_atomic());
        let Some(data_race) = this.machine.data_race.as_vclocks_ref() else {
            return interp_ok(());
        };
        if !data_race.race_detecting() {
            return interp_ok(());
        }
        let size = place.layout.size;
        let (alloc_id, base_offset, _prov) = this.ptr_get_alloc_id(place.ptr(), 0)?;
        // Load and log the atomic operation.
        // Note that atomic loads are possible even from read-only allocations, so `get_alloc_extra_mut` is not an option.
        let alloc_meta = this.get_alloc_extra(alloc_id)?.data_race.as_vclocks_ref().unwrap();
        trace!(
            "Atomic op({}) with ordering {:?} on {:?} (size={})",
            access.description(None, None),
            &atomic,
            place.ptr(),
            size.bytes()
        );

        let current_span = this.machine.current_span();
        // Perform the atomic operation.
        data_race.maybe_perform_sync_operation(
            &this.machine.threads,
            current_span,
            |index, mut thread_clocks| {
                for (mem_clocks_range, mem_clocks) in
                    alloc_meta.alloc_ranges.borrow_mut().iter_mut(base_offset, size)
                {
                    if let Err(DataRace) = op(mem_clocks, &mut thread_clocks, index, atomic) {
                        mem::drop(thread_clocks);
                        return VClockAlloc::report_data_race(
                            data_race,
                            &this.machine.threads,
                            mem_clocks,
                            access,
                            place.layout.size,
                            interpret::Pointer::new(
                                alloc_id,
                                Size::from_bytes(mem_clocks_range.start),
                            ),
                            None,
                        )
                        .map(|_| true);
                    }
                }

                // This conservatively assumes all operations have release semantics
                interp_ok(true)
            },
        )?;

        // Log changes to atomic memory.
        if tracing::enabled!(tracing::Level::TRACE) {
            for (_offset, mem_clocks) in alloc_meta.alloc_ranges.borrow().iter(base_offset, size) {
                trace!(
                    "Updated atomic memory({:?}, size={}) to {:#?}",
                    place.ptr(),
                    size.bytes(),
                    mem_clocks.atomic_ops
                );
            }
        }

        interp_ok(())
    }
}

impl GlobalState {
    /// Create a new global state, setup with just thread-id=0
    /// advanced to timestamp = 1.
    pub fn new(config: &MiriConfig) -> Self {
        let mut global_state = GlobalState {
            multi_threaded: Cell::new(false),
            ongoing_action_data_race_free: Cell::new(false),
            vector_clocks: RefCell::new(IndexVec::new()),
            vector_info: RefCell::new(IndexVec::new()),
            thread_info: RefCell::new(IndexVec::new()),
            reuse_candidates: RefCell::new(FxHashSet::default()),
            last_sc_fence: RefCell::new(VClock::default()),
            last_sc_write_per_thread: RefCell::new(VClock::default()),
            track_outdated_loads: config.track_outdated_loads,
            weak_memory: config.weak_memory_emulation,
        };

        // Setup the main-thread since it is not explicitly created:
        // uses vector index and thread-id 0.
        let index = global_state.vector_clocks.get_mut().push(ThreadClockSet::default());
        global_state.vector_info.get_mut().push(ThreadId::MAIN_THREAD);
        global_state
            .thread_info
            .get_mut()
            .push(ThreadExtraState { vector_index: Some(index), termination_vector_clock: None });

        global_state
    }

    // We perform data race detection when there are more than 1 active thread
    // and we have not temporarily disabled race detection to perform something
    // data race free
    fn race_detecting(&self) -> bool {
        self.multi_threaded.get() && !self.ongoing_action_data_race_free.get()
    }

    pub fn ongoing_action_data_race_free(&self) -> bool {
        self.ongoing_action_data_race_free.get()
    }

    // Try to find vector index values that can potentially be re-used
    // by a new thread instead of a new vector index being created.
    fn find_vector_index_reuse_candidate(&self) -> Option<VectorIdx> {
        let mut reuse = self.reuse_candidates.borrow_mut();
        let vector_clocks = self.vector_clocks.borrow();
        for &candidate in reuse.iter() {
            let target_timestamp = vector_clocks[candidate].clock[candidate];
            if vector_clocks.iter_enumerated().all(|(clock_idx, clock)| {
                // The thread happens before the clock, and hence cannot report
                // a data-race with this the candidate index.
                let no_data_race = clock.clock[candidate] >= target_timestamp;

                // The vector represents a thread that has terminated and hence cannot
                // report a data-race with the candidate index.
                let vector_terminated = reuse.contains(&clock_idx);

                // The vector index cannot report a race with the candidate index
                // and hence allows the candidate index to be re-used.
                no_data_race || vector_terminated
            }) {
                // All vector clocks for each vector index are equal to
                // the target timestamp, and the thread is known to have
                // terminated, therefore this vector clock index cannot
                // report any more data-races.
                assert!(reuse.remove(&candidate));
                return Some(candidate);
            }
        }
        None
    }

    // Hook for thread creation, enabled multi-threaded execution and marks
    // the current thread timestamp as happening-before the current thread.
    #[inline]
    pub fn thread_created(
        &mut self,
        thread_mgr: &ThreadManager<'_>,
        thread: ThreadId,
        current_span: Span,
    ) {
        let current_index = self.active_thread_index(thread_mgr);

        // Enable multi-threaded execution, there are now at least two threads
        // so data-races are now possible.
        self.multi_threaded.set(true);

        // Load and setup the associated thread metadata
        let mut thread_info = self.thread_info.borrow_mut();
        thread_info.ensure_contains_elem(thread, Default::default);

        // Assign a vector index for the thread, attempting to re-use an old
        // vector index that can no longer report any data-races if possible.
        let created_index = if let Some(reuse_index) = self.find_vector_index_reuse_candidate() {
            // Now re-configure the re-use candidate, increment the clock
            // for the new sync use of the vector.
            let vector_clocks = self.vector_clocks.get_mut();
            vector_clocks[reuse_index].increment_clock(reuse_index, current_span);

            // Locate the old thread the vector was associated with and update
            // it to represent the new thread instead.
            let vector_info = self.vector_info.get_mut();
            let old_thread = vector_info[reuse_index];
            vector_info[reuse_index] = thread;

            // Mark the thread the vector index was associated with as no longer
            // representing a thread index.
            thread_info[old_thread].vector_index = None;

            reuse_index
        } else {
            // No vector re-use candidates available, instead create
            // a new vector index.
            let vector_info = self.vector_info.get_mut();
            vector_info.push(thread)
        };

        trace!("Creating thread = {:?} with vector index = {:?}", thread, created_index);

        // Mark the chosen vector index as in use by the thread.
        thread_info[thread].vector_index = Some(created_index);

        // Create a thread clock set if applicable.
        let vector_clocks = self.vector_clocks.get_mut();
        if created_index == vector_clocks.next_index() {
            vector_clocks.push(ThreadClockSet::default());
        }

        // Now load the two clocks and configure the initial state.
        let (current, created) = vector_clocks.pick2_mut(current_index, created_index);

        // Join the created with current, since the current threads
        // previous actions happen-before the created thread.
        created.join_with(current);

        // Advance both threads after the synchronized operation.
        // Both operations are considered to have release semantics.
        current.increment_clock(current_index, current_span);
        created.increment_clock(created_index, current_span);
    }

    /// Hook on a thread join to update the implicit happens-before relation between the joined
    /// thread (the joinee, the thread that someone waited on) and the current thread (the joiner,
    /// the thread who was waiting).
    #[inline]
    pub fn thread_joined(&mut self, threads: &ThreadManager<'_>, joinee: ThreadId) {
        let thread_info = self.thread_info.borrow();
        let thread_info = &thread_info[joinee];

        // Load the associated vector clock for the terminated thread.
        let join_clock = thread_info
            .termination_vector_clock
            .as_ref()
            .expect("joined with thread but thread has not terminated");
        // Acquire that into the current thread.
        self.acquire_clock(join_clock, threads);

        // Check the number of live threads, if the value is 1
        // then test for potentially disabling multi-threaded execution.
        // This has to happen after `acquire_clock`, otherwise there'll always
        // be some thread that has not synchronized yet.
        if let Some(current_index) = thread_info.vector_index {
            if threads.get_live_thread_count() == 1 {
                let vector_clocks = self.vector_clocks.get_mut();
                // May potentially be able to disable multi-threaded execution.
                let current_clock = &vector_clocks[current_index];
                if vector_clocks
                    .iter_enumerated()
                    .all(|(idx, clocks)| clocks.clock[idx] <= current_clock.clock[idx])
                {
                    // All thread terminations happen-before the current clock
                    // therefore no data-races can be reported until a new thread
                    // is created, so disable multi-threaded execution.
                    self.multi_threaded.set(false);
                }
            }
        }
    }

    /// On thread termination, the vector clock may be re-used
    /// in the future once all remaining thread-clocks catch
    /// up with the time index of the terminated thread.
    /// This assigns thread termination with a unique index
    /// which will be used to join the thread
    /// This should be called strictly before any calls to
    /// `thread_joined`.
    #[inline]
    pub fn thread_terminated(&mut self, thread_mgr: &ThreadManager<'_>) {
        let current_thread = thread_mgr.active_thread();
        let current_index = self.active_thread_index(thread_mgr);

        // Store the terminaion clock.
        let terminaion_clock = self.release_clock(thread_mgr, |clock| clock.clone());
        self.thread_info.get_mut()[current_thread].termination_vector_clock =
            Some(terminaion_clock);

        // Add this thread's clock index as a candidate for re-use.
        let reuse = self.reuse_candidates.get_mut();
        reuse.insert(current_index);
    }

    /// Update the data-race detector for an atomic fence on the current thread.
    fn atomic_fence<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let current_span = machine.current_span();
        self.maybe_perform_sync_operation(&machine.threads, current_span, |index, mut clocks| {
            trace!("Atomic fence on {:?} with ordering {:?}", index, atomic);

            // Apply data-race detection for the current fences
            // this treats AcqRel and SeqCst as the same as an acquire
            // and release fence applied in the same timestamp.
            if atomic != AtomicFenceOrd::Release {
                // Either Acquire | AcqRel | SeqCst
                clocks.apply_acquire_fence();
            }
            if atomic == AtomicFenceOrd::SeqCst {
                // Behave like an RMW on the global fence location. This takes full care of
                // all the SC fence requirements, including C++17 §32.4 [atomics.order]
                // paragraph 6 (which would limit what future reads can see). It also rules
                // out many legal behaviors, but we don't currently have a model that would
                // be more precise.
                // Also see the second bullet on page 10 of
                // <https://www.cs.tau.ac.il/~orilahav/papers/popl21_robustness.pdf>.
                let mut sc_fence_clock = self.last_sc_fence.borrow_mut();
                sc_fence_clock.join(&clocks.clock);
                clocks.clock.join(&sc_fence_clock);
                // Also establish some sort of order with the last SC write that happened, globally
                // (but this is only respected by future reads).
                clocks.write_seqcst.join(&self.last_sc_write_per_thread.borrow());
            }
            // The release fence is last, since both of the above could alter our clock,
            // which should be part of what is being released.
            if atomic != AtomicFenceOrd::Acquire {
                // Either Release | AcqRel | SeqCst
                clocks.apply_release_fence();
            }

            // Increment timestamp in case of release semantics.
            interp_ok(atomic != AtomicFenceOrd::Acquire)
        })
    }

    /// Attempt to perform a synchronized operation, this
    /// will perform no operation if multi-threading is
    /// not currently enabled.
    /// Otherwise it will increment the clock for the current
    /// vector before and after the operation for data-race
    /// detection between any happens-before edges the
    /// operation may create.
    fn maybe_perform_sync_operation<'tcx>(
        &self,
        thread_mgr: &ThreadManager<'_>,
        current_span: Span,
        op: impl FnOnce(VectorIdx, RefMut<'_, ThreadClockSet>) -> InterpResult<'tcx, bool>,
    ) -> InterpResult<'tcx> {
        if self.multi_threaded.get() {
            let (index, clocks) = self.active_thread_state_mut(thread_mgr);
            if op(index, clocks)? {
                let (_, mut clocks) = self.active_thread_state_mut(thread_mgr);
                clocks.increment_clock(index, current_span);
            }
        }
        interp_ok(())
    }

    /// Internal utility to identify a thread stored internally
    /// returns the id and the name for better diagnostics.
    fn print_thread_metadata(&self, thread_mgr: &ThreadManager<'_>, vector: VectorIdx) -> String {
        let thread = self.vector_info.borrow()[vector];
        let thread_name = thread_mgr.get_thread_display_name(thread);
        format!("thread `{thread_name}`")
    }

    /// Acquire the given clock into the current thread, establishing synchronization with
    /// the moment when that clock snapshot was taken via `release_clock`.
    /// As this is an acquire operation, the thread timestamp is not
    /// incremented.
    pub fn acquire_clock<'tcx>(&self, clock: &VClock, threads: &ThreadManager<'tcx>) {
        let thread = threads.active_thread();
        let (_, mut clocks) = self.thread_state_mut(thread);
        clocks.clock.join(clock);
    }

    /// Calls the given closure with the "release" clock of the current thread.
    /// Other threads can acquire this clock in the future to establish synchronization
    /// with this program point.
    pub fn release_clock<'tcx, R>(
        &self,
        threads: &ThreadManager<'tcx>,
        callback: impl FnOnce(&VClock) -> R,
    ) -> R {
        let thread = threads.active_thread();
        let span = threads.active_thread_ref().current_span();
        let (index, mut clocks) = self.thread_state_mut(thread);
        let r = callback(&clocks.clock);
        // Increment the clock, so that all following events cannot be confused with anything that
        // occurred before the release. Crucially, the callback is invoked on the *old* clock!
        clocks.increment_clock(index, span);

        r
    }

    fn thread_index(&self, thread: ThreadId) -> VectorIdx {
        self.thread_info.borrow()[thread].vector_index.expect("thread has no assigned vector")
    }

    /// Load the vector index used by the given thread as well as the set of vector clocks
    /// used by the thread.
    #[inline]
    fn thread_state_mut(&self, thread: ThreadId) -> (VectorIdx, RefMut<'_, ThreadClockSet>) {
        let index = self.thread_index(thread);
        let ref_vector = self.vector_clocks.borrow_mut();
        let clocks = RefMut::map(ref_vector, |vec| &mut vec[index]);
        (index, clocks)
    }

    /// Load the vector index used by the given thread as well as the set of vector clocks
    /// used by the thread.
    #[inline]
    fn thread_state(&self, thread: ThreadId) -> (VectorIdx, Ref<'_, ThreadClockSet>) {
        let index = self.thread_index(thread);
        let ref_vector = self.vector_clocks.borrow();
        let clocks = Ref::map(ref_vector, |vec| &vec[index]);
        (index, clocks)
    }

    /// Load the current vector clock in use and the current set of thread clocks
    /// in use for the vector.
    #[inline]
    pub(super) fn active_thread_state(
        &self,
        thread_mgr: &ThreadManager<'_>,
    ) -> (VectorIdx, Ref<'_, ThreadClockSet>) {
        self.thread_state(thread_mgr.active_thread())
    }

    /// Load the current vector clock in use and the current set of thread clocks
    /// in use for the vector mutably for modification.
    #[inline]
    pub(super) fn active_thread_state_mut(
        &self,
        thread_mgr: &ThreadManager<'_>,
    ) -> (VectorIdx, RefMut<'_, ThreadClockSet>) {
        self.thread_state_mut(thread_mgr.active_thread())
    }

    /// Return the current thread, should be the same
    /// as the data-race active thread.
    #[inline]
    fn active_thread_index(&self, thread_mgr: &ThreadManager<'_>) -> VectorIdx {
        let active_thread_id = thread_mgr.active_thread();
        self.thread_index(active_thread_id)
    }

    // SC ATOMIC STORE rule in the paper.
    pub(super) fn sc_write(&self, thread_mgr: &ThreadManager<'_>) {
        let (index, clocks) = self.active_thread_state(thread_mgr);
        self.last_sc_write_per_thread.borrow_mut().set_at_index(&clocks.clock, index);
    }

    // SC ATOMIC READ rule in the paper.
    pub(super) fn sc_read(&self, thread_mgr: &ThreadManager<'_>) {
        let (.., mut clocks) = self.active_thread_state_mut(thread_mgr);
        clocks.read_seqcst.join(&self.last_sc_fence.borrow());
    }
}
