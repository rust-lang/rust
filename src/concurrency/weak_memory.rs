//! Implementation of C++11-consistent weak memory emulation using store buffers
//! based on Dynamic Race Detection for C++ ("the paper"):
//! <https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf>
//!
//! This implementation will never generate weak memory behaviours forbidden by the C++11 model,
//! but it is incapable of producing all possible weak behaviours allowed by the model. There are
//! certain weak behaviours observable on real hardware but not while using this.
//!
//! Note that this implementation does not take into account of C++20's memory model revision to SC accesses
//! and fences introduced by P0668 (<https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0668r5.html>).
//! This implementation is not fully correct under the revised C++20 model and may generate behaviours C++20
//! disallows (<https://github.com/rust-lang/miri/issues/2301>).
//!
//! Rust follows the C++20 memory model (except for the Consume ordering and some operations not performable through C++'s
//! std::atomic<T> API). It is therefore possible for this implementation to generate behaviours never observable when the
//! same program is compiled and run natively. Unfortunately, no literature exists at the time of writing which proposes
//! an implementable and C++20-compatible relaxed memory model that supports all atomic operation existing in Rust. The closest one is
//! A Promising Semantics for Relaxed-Memory Concurrency by Jeehoon Kang et al. (<https://www.cs.tau.ac.il/~orilahav/papers/popl17.pdf>)
//! However, this model lacks SC accesses and is therefore unusable by Miri (SC accesses are everywhere in library code).
//!
//! If you find anything that proposes a relaxed memory model that is C++20-consistent, supports all orderings Rust's atomic accesses
//! and fences accept, and is implementable (with operational semanitcs), please open a GitHub issue!
//!
//! One characteristic of this implementation, in contrast to some other notable operational models such as ones proposed in
//! Taming Release-Acquire Consistency by Ori Lahav et al. (<https://plv.mpi-sws.org/sra/paper.pdf>) or Promising Semantics noted above,
//! is that this implementation does not require each thread to hold an isolated view of the entire memory. Here, store buffers are per-location
//! and shared across all threads. This is more memory efficient but does require store elements (representing writes to a location) to record
//! information about reads, whereas in the other two models it is the other way round: reads points to the write it got its value from.
//! Additionally, writes in our implementation do not have globally unique timestamps attached. In the other two models this timestamp is
//! used to make sure a value in a thread's view is not overwritten by a write that occured earlier than the one in the existing view.
//! In our implementation, this is detected using read information attached to store elements, as there is no data strucutre representing reads.
//!
//! The C++ memory model is built around the notion of an 'atomic object', so it would be natural
//! to attach store buffers to atomic objects. However, Rust follows LLVM in that it only has
//! 'atomic accesses'. Therefore Miri cannot know when and where atomic 'objects' are being
//! created or destroyed, to manage its store buffers. Instead, we hence lazily create an
//! atomic object on the first atomic access to a given region, and we destroy that object
//! on the next non-atomic or imperfectly overlapping atomic access to that region.
//! These lazy (de)allocations happen in memory_accessed() on non-atomic accesses, and
//! get_or_create_store_buffer() on atomic accesses. This mostly works well, but it does
//! lead to some issues (<https://github.com/rust-lang/miri/issues/2164>).
//!
//! One consequence of this difference is that safe/sound Rust allows for more operations on atomic locations
//! than the C++20 atomic API was intended to allow, such as non-atomically accessing
//! a previously atomically accessed location, or accessing previously atomically accessed locations with a differently sized operation
//! (such as accessing the top 16 bits of an AtomicU32). These senarios are generally undiscussed in formalisations of C++ memory model.
//! In Rust, these operations can only be done through a `&mut AtomicFoo` reference or one derived from it, therefore these operations
//! can only happen after all previous accesses on the same locations. This implementation is adapted to allow these operations.
//! A mixed atomicity read that races with writes, or a write that races with reads or writes will still cause UBs to be thrown.
//! Mixed size atomic accesses must not race with any other atomic access, whether read or write, or a UB will be thrown.
//! You can refer to test cases in weak_memory/extra_cpp.rs and weak_memory/extra_cpp_unsafe.rs for examples of these operations.

// Our and the author's own implementation (tsan11) of the paper have some deviations from the provided operational semantics in §5.3:
// 1. In the operational semantics, store elements keep a copy of the atomic object's vector clock (AtomicCellClocks::sync_vector in miri),
// but this is not used anywhere so it's omitted here.
//
// 2. In the operational semantics, each store element keeps the timestamp of a thread when it loads from the store.
// If the same thread loads from the same store element multiple times, then the timestamps at all loads are saved in a list of load elements.
// This is not necessary as later loads by the same thread will always have greater timetstamp values, so we only need to record the timestamp of the first
// load by each thread. This optimisation is done in tsan11
// (https://github.com/ChrisLidbury/tsan11/blob/ecbd6b81e9b9454e01cba78eb9d88684168132c7/lib/tsan/rtl/tsan_relaxed.h#L35-L37)
// and here.
//
// 3. §4.5 of the paper wants an SC store to mark all existing stores in the buffer that happens before it
// as SC. This is not done in the operational semantics but implemented correctly in tsan11
// (https://github.com/ChrisLidbury/tsan11/blob/ecbd6b81e9b9454e01cba78eb9d88684168132c7/lib/tsan/rtl/tsan_relaxed.cc#L160-L167)
// and here.
//
// 4. W_SC ; R_SC case requires the SC load to ignore all but last store maked SC (stores not marked SC are not
// affected). But this rule is applied to all loads in ReadsFromSet from the paper (last two lines of code), not just SC load.
// This is implemented correctly in tsan11
// (https://github.com/ChrisLidbury/tsan11/blob/ecbd6b81e9b9454e01cba78eb9d88684168132c7/lib/tsan/rtl/tsan_relaxed.cc#L295)
// and here.

use std::{
    cell::{Ref, RefCell},
    collections::VecDeque,
};

use rustc_const_eval::interpret::{
    alloc_range, AllocRange, InterpResult, MPlaceTy, ScalarMaybeUninit,
};
use rustc_data_structures::fx::FxHashMap;

use crate::{
    AtomicReadOrd, AtomicRwOrd, AtomicWriteOrd, Tag, ThreadManager, VClock, VTimestamp, VectorIdx,
};

use super::{
    data_race::{GlobalState as DataRaceState, ThreadClockSet},
    range_object_map::{AccessType, RangeObjectMap},
};

pub type AllocExtra = StoreBufferAlloc;

// Each store buffer must be bounded otherwise it will grow indefinitely.
// However, bounding the store buffer means restricting the amount of weak
// behaviours observable. The author picked 128 as a good tradeoff
// so we follow them here.
const STORE_BUFFER_LIMIT: usize = 128;

#[derive(Debug, Clone)]
pub struct StoreBufferAlloc {
    /// Store buffer of each atomic object in this allocation
    // Behind a RefCell because we need to allocate/remove on read access
    store_buffers: RefCell<RangeObjectMap<StoreBuffer>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StoreBuffer {
    // Stores to this location in modification order
    buffer: VecDeque<StoreElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StoreElement {
    /// The identifier of the vector index, corresponding to a thread
    /// that performed the store.
    store_index: VectorIdx,

    /// Whether this store is SC.
    is_seqcst: bool,

    /// The timestamp of the storing thread when it performed the store
    timestamp: VTimestamp,
    /// The value of this store
    // FIXME: this means the store is either fully initialized or fully uninitialized;
    // we will have to change this if we want to support atomics on
    // partially initialized data.
    val: ScalarMaybeUninit<Tag>,

    /// Timestamp of first loads from this store element by each thread
    /// Behind a RefCell to keep load op take &self
    loads: RefCell<FxHashMap<VectorIdx, VTimestamp>>,
}

impl StoreBufferAlloc {
    pub fn new_allocation() -> Self {
        Self { store_buffers: RefCell::new(RangeObjectMap::new()) }
    }

    /// Checks if the range imperfectly overlaps with existing buffers
    /// Used to determine if mixed-size atomic accesses
    fn is_overlapping(&self, range: AllocRange) -> bool {
        let buffers = self.store_buffers.borrow();
        let access_type = buffers.access_type(range);
        matches!(access_type, AccessType::ImperfectlyOverlapping(_))
    }

    /// When a non-atomic access happens on a location that has been atomically accessed
    /// before without data race, we can determine that the non-atomic access fully happens
    /// after all the prior atomic accesses so the location no longer needs to exhibit
    /// any weak memory behaviours until further atomic accesses.
    pub fn memory_accessed(&self, range: AllocRange, global: &DataRaceState) {
        if !global.ongoing_action_data_race_free() {
            let mut buffers = self.store_buffers.borrow_mut();
            let access_type = buffers.access_type(range);
            match access_type {
                AccessType::PerfectlyOverlapping(pos) => {
                    buffers.remove_from_pos(pos);
                }
                AccessType::ImperfectlyOverlapping(pos_range) => {
                    buffers.remove_pos_range(pos_range);
                }
                AccessType::Empty(_) => {
                    // The range had no weak behaivours attached, do nothing
                }
            }
        }
    }

    /// Gets a store buffer associated with an atomic object in this allocation,
    /// or creates one with the specified initial value if no atomic object exists yet.
    fn get_or_create_store_buffer<'tcx>(
        &self,
        range: AllocRange,
        init: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx, Ref<'_, StoreBuffer>> {
        let access_type = self.store_buffers.borrow().access_type(range);
        let pos = match access_type {
            AccessType::PerfectlyOverlapping(pos) => pos,
            AccessType::Empty(pos) => {
                let mut buffers = self.store_buffers.borrow_mut();
                buffers.insert_at_pos(pos, range, StoreBuffer::new(init));
                pos
            }
            AccessType::ImperfectlyOverlapping(pos_range) => {
                // Once we reach here we would've already checked that this access is not racy
                let mut buffers = self.store_buffers.borrow_mut();
                buffers.remove_pos_range(pos_range.clone());
                buffers.insert_at_pos(pos_range.start, range, StoreBuffer::new(init));
                pos_range.start
            }
        };
        Ok(Ref::map(self.store_buffers.borrow(), |buffer| &buffer[pos]))
    }

    /// Gets a mutable store buffer associated with an atomic object in this allocation
    fn get_or_create_store_buffer_mut<'tcx>(
        &mut self,
        range: AllocRange,
        init: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx, &mut StoreBuffer> {
        let buffers = self.store_buffers.get_mut();
        let access_type = buffers.access_type(range);
        let pos = match access_type {
            AccessType::PerfectlyOverlapping(pos) => pos,
            AccessType::Empty(pos) => {
                buffers.insert_at_pos(pos, range, StoreBuffer::new(init));
                pos
            }
            AccessType::ImperfectlyOverlapping(pos_range) => {
                buffers.remove_pos_range(pos_range.clone());
                buffers.insert_at_pos(pos_range.start, range, StoreBuffer::new(init));
                pos_range.start
            }
        };
        Ok(&mut buffers[pos])
    }
}

impl<'mir, 'tcx: 'mir> StoreBuffer {
    fn new(init: ScalarMaybeUninit<Tag>) -> Self {
        let mut buffer = VecDeque::new();
        buffer.reserve(STORE_BUFFER_LIMIT);
        let mut ret = Self { buffer };
        let store_elem = StoreElement {
            // The thread index and timestamp of the initialisation write
            // are never meaningfully used, so it's fine to leave them as 0
            store_index: VectorIdx::from(0),
            timestamp: 0,
            val: init,
            is_seqcst: false,
            loads: RefCell::new(FxHashMap::default()),
        };
        ret.buffer.push_back(store_elem);
        ret
    }

    /// Reads from the last store in modification order
    fn read_from_last_store(&self, global: &DataRaceState, thread_mgr: &ThreadManager<'_, '_>) {
        let store_elem = self.buffer.back();
        if let Some(store_elem) = store_elem {
            let (index, clocks) = global.current_thread_state(thread_mgr);
            store_elem.load_impl(index, &clocks);
        }
    }

    fn buffered_read(
        &self,
        global: &DataRaceState,
        thread_mgr: &ThreadManager<'_, '_>,
        is_seqcst: bool,
        rng: &mut (impl rand::Rng + ?Sized),
        validate: impl FnOnce() -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        // Having a live borrow to store_buffer while calling validate_atomic_load is fine
        // because the race detector doesn't touch store_buffer

        let store_elem = {
            // The `clocks` we got here must be dropped before calling validate_atomic_load
            // as the race detector will update it
            let (.., clocks) = global.current_thread_state(thread_mgr);
            // Load from a valid entry in the store buffer
            self.fetch_store(is_seqcst, &clocks, &mut *rng)
        };

        // Unlike in buffered_atomic_write, thread clock updates have to be done
        // after we've picked a store element from the store buffer, as presented
        // in ATOMIC LOAD rule of the paper. This is because fetch_store
        // requires access to ThreadClockSet.clock, which is updated by the race detector
        validate()?;

        let (index, clocks) = global.current_thread_state(thread_mgr);
        let loaded = store_elem.load_impl(index, &clocks);
        Ok(loaded)
    }

    fn buffered_write(
        &mut self,
        val: ScalarMaybeUninit<Tag>,
        global: &DataRaceState,
        thread_mgr: &ThreadManager<'_, '_>,
        is_seqcst: bool,
    ) -> InterpResult<'tcx> {
        let (index, clocks) = global.current_thread_state(thread_mgr);

        self.store_impl(val, index, &clocks.clock, is_seqcst);
        Ok(())
    }

    /// Selects a valid store element in the buffer.
    fn fetch_store<R: rand::Rng + ?Sized>(
        &self,
        is_seqcst: bool,
        clocks: &ThreadClockSet,
        rng: &mut R,
    ) -> &StoreElement {
        use rand::seq::IteratorRandom;
        let mut found_sc = false;
        // FIXME: we want an inclusive take_while (stops after a false predicate, but
        // includes the element that gave the false), but such function doesn't yet
        // exist in the standard libary https://github.com/rust-lang/rust/issues/62208
        // so we have to hack around it with keep_searching
        let mut keep_searching = true;
        let candidates = self
            .buffer
            .iter()
            .rev()
            .take_while(move |&store_elem| {
                if !keep_searching {
                    return false;
                }

                keep_searching = if store_elem.timestamp <= clocks.clock[store_elem.store_index] {
                    // CoWR: if a store happens-before the current load,
                    // then we can't read-from anything earlier in modification order.
                    log::info!("Stopping due to coherent write-read");
                    false
                } else if store_elem.loads.borrow().iter().any(|(&load_index, &load_timestamp)| {
                    load_timestamp <= clocks.clock[load_index]
                }) {
                    // CoRR: if there was a load from this store which happened-before the current load,
                    // then we cannot read-from anything earlier in modification order.
                    log::info!("Stopping due to coherent read-read");
                    false
                } else if store_elem.timestamp <= clocks.fence_seqcst[store_elem.store_index] {
                    // The current load, which may be sequenced-after an SC fence, can only read-from
                    // the last store sequenced-before an SC fence in another thread (or any stores
                    // later than that SC fence)
                    log::info!("Stopping due to coherent load sequenced after sc fence");
                    false
                } else if store_elem.timestamp <= clocks.write_seqcst[store_elem.store_index]
                    && store_elem.is_seqcst
                {
                    // The current non-SC load can only read-from the latest SC store (or any stores later than that
                    // SC store)
                    log::info!("Stopping due to needing to load from the last SC store");
                    false
                } else if is_seqcst && store_elem.timestamp <= clocks.read_seqcst[store_elem.store_index] {
                    // The current SC load can only read-from the last store sequenced-before
                    // the last SC fence (or any stores later than the SC fence)
                    log::info!("Stopping due to sc load needing to load from the last SC store before an SC fence");
                    false
                } else {true};

                true
            })
            .filter(|&store_elem| {
                if is_seqcst && store_elem.is_seqcst {
                    // An SC load needs to ignore all but last store maked SC (stores not marked SC are not
                    // affected)
                    let include = !found_sc;
                    found_sc = true;
                    include
                } else {
                    true
                }
            });

        candidates
            .choose(rng)
            .expect("store buffer cannot be empty, an element is populated on construction")
    }

    /// ATOMIC STORE IMPL in the paper (except we don't need the location's vector clock)
    fn store_impl(
        &mut self,
        val: ScalarMaybeUninit<Tag>,
        index: VectorIdx,
        thread_clock: &VClock,
        is_seqcst: bool,
    ) {
        let store_elem = StoreElement {
            store_index: index,
            timestamp: thread_clock[index],
            // In the language provided in the paper, an atomic store takes the value from a
            // non-atomic memory location.
            // But we already have the immediate value here so we don't need to do the memory
            // access
            val,
            is_seqcst,
            loads: RefCell::new(FxHashMap::default()),
        };
        self.buffer.push_back(store_elem);
        if self.buffer.len() > STORE_BUFFER_LIMIT {
            self.buffer.pop_front();
        }
        if is_seqcst {
            // Every store that happens before this needs to be marked as SC
            // so that in a later SC load, only the last SC store (i.e. this one) or stores that
            // aren't ordered by hb with the last SC is picked.
            self.buffer.iter_mut().rev().for_each(|elem| {
                if elem.timestamp <= thread_clock[elem.store_index] {
                    elem.is_seqcst = true;
                }
            })
        }
    }
}

impl StoreElement {
    /// ATOMIC LOAD IMPL in the paper
    /// Unlike the operational semantics in the paper, we don't need to keep track
    /// of the thread timestamp for every single load. Keeping track of the first (smallest)
    /// timestamp of each thread that has loaded from a store is sufficient: if the earliest
    /// load of another thread happens before the current one, then we must stop searching the store
    /// buffer regardless of subsequent loads by the same thread; if the earliest load of another
    /// thread doesn't happen before the current one, then no subsequent load by the other thread
    /// can happen before the current one.
    fn load_impl(&self, index: VectorIdx, clocks: &ThreadClockSet) -> ScalarMaybeUninit<Tag> {
        let _ = self.loads.borrow_mut().try_insert(index, clocks.clock[index]);
        self.val
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub(super) trait EvalContextExt<'mir, 'tcx: 'mir>:
    crate::MiriEvalContextExt<'mir, 'tcx>
{
    // If weak memory emulation is enabled, check if this atomic op imperfectly overlaps with a previous
    // atomic read or write. If it does, then we require it to be ordered (non-racy) with all previous atomic
    // accesses on all the bytes in range
    fn validate_overlapping_atomic(&self, place: &MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
        if let crate::AllocExtra {
            weak_memory: Some(alloc_buffers),
            data_race: Some(alloc_clocks),
            ..
        } = this.get_alloc_extra(alloc_id)?
        {
            let range = alloc_range(base_offset, place.layout.size);
            if alloc_buffers.is_overlapping(range)
                && !alloc_clocks.race_free_with_atomic(
                    range,
                    this.machine.data_race.as_ref().unwrap(),
                    &this.machine.threads,
                )
            {
                throw_unsup_format!(
                    "racy imperfectly overlapping atomic access is not possible in the C++20 memory model, and not supported by Miri's weak memory emulation"
                );
            }
        }
        Ok(())
    }

    fn buffered_atomic_rmw(
        &mut self,
        new_val: ScalarMaybeUninit<Tag>,
        place: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicRwOrd,
        init: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
        if let (
            crate::AllocExtra { weak_memory: Some(alloc_buffers), .. },
            crate::Evaluator { data_race: Some(global), threads, .. },
        ) = this.get_alloc_extra_mut(alloc_id)?
        {
            if atomic == AtomicRwOrd::SeqCst {
                global.sc_read(threads);
                global.sc_write(threads);
            }
            let range = alloc_range(base_offset, place.layout.size);
            let buffer = alloc_buffers.get_or_create_store_buffer_mut(range, init)?;
            buffer.read_from_last_store(global, threads);
            buffer.buffered_write(new_val, global, threads, atomic == AtomicRwOrd::SeqCst)?;
        }
        Ok(())
    }

    fn buffered_atomic_read(
        &self,
        place: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicReadOrd,
        latest_in_mo: ScalarMaybeUninit<Tag>,
        validate: impl FnOnce() -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_ref();
        if let Some(global) = &this.machine.data_race {
            let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
            if let Some(alloc_buffers) = this.get_alloc_extra(alloc_id)?.weak_memory.as_ref() {
                if atomic == AtomicReadOrd::SeqCst {
                    global.sc_read(&this.machine.threads);
                }
                let mut rng = this.machine.rng.borrow_mut();
                let buffer = alloc_buffers.get_or_create_store_buffer(
                    alloc_range(base_offset, place.layout.size),
                    latest_in_mo,
                )?;
                let loaded = buffer.buffered_read(
                    global,
                    &this.machine.threads,
                    atomic == AtomicReadOrd::SeqCst,
                    &mut *rng,
                    validate,
                )?;

                return Ok(loaded);
            }
        }

        // Race detector or weak memory disabled, simply read the latest value
        validate()?;
        Ok(latest_in_mo)
    }

    fn buffered_atomic_write(
        &mut self,
        val: ScalarMaybeUninit<Tag>,
        dest: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicWriteOrd,
        init: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(dest.ptr)?;
        if let (
            crate::AllocExtra { weak_memory: Some(alloc_buffers), .. },
            crate::Evaluator { data_race: Some(global), threads, .. },
        ) = this.get_alloc_extra_mut(alloc_id)?
        {
            if atomic == AtomicWriteOrd::SeqCst {
                global.sc_write(threads);
            }

            // UGLY HACK: in write_scalar_atomic() we don't know the value before our write,
            // so init == val always. If the buffer is fresh then we would've duplicated an entry,
            // so we need to remove it.
            // See https://github.com/rust-lang/miri/issues/2164
            let was_empty = matches!(
                alloc_buffers
                    .store_buffers
                    .borrow()
                    .access_type(alloc_range(base_offset, dest.layout.size)),
                AccessType::Empty(_)
            );
            let buffer = alloc_buffers
                .get_or_create_store_buffer_mut(alloc_range(base_offset, dest.layout.size), init)?;
            if was_empty {
                buffer.buffer.pop_front();
            }

            buffer.buffered_write(val, global, threads, atomic == AtomicWriteOrd::SeqCst)?;
        }

        // Caller should've written to dest with the vanilla scalar write, we do nothing here
        Ok(())
    }

    /// Caller should never need to consult the store buffer for the latest value.
    /// This function is used exclusively for failed atomic_compare_exchange_scalar
    /// to perform load_impl on the latest store element
    fn perform_read_on_buffered_latest(
        &self,
        place: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicReadOrd,
        init: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();

        if let Some(global) = &this.machine.data_race {
            if atomic == AtomicReadOrd::SeqCst {
                global.sc_read(&this.machine.threads);
            }
            let size = place.layout.size;
            let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
            if let Some(alloc_buffers) = this.get_alloc_extra(alloc_id)?.weak_memory.as_ref() {
                let buffer = alloc_buffers
                    .get_or_create_store_buffer(alloc_range(base_offset, size), init)?;
                buffer.read_from_last_store(global, &this.machine.threads);
            }
        }
        Ok(())
    }
}
