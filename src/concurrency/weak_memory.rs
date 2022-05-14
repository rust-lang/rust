//! Implementation of C++11-consistent weak memory emulation using store buffers
//! based on Dynamic Race Detection for C++ ("the paper"):
//! https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf
//!
//! This implementation will never generate weak memory behaviours forbidden by the C++11 model,
//! but it is incapable of producing all possible weak behaviours allowed by the model. There are
//! certain weak behaviours observable on real hardware but not while using this.
//!
//! Note that this implementation does not take into account of C++20's memory model revision to SC accesses
//! and fences introduced by P0668 (https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0668r5.html).
//! This implementation is not fully correct under the revised C++20 model and may generate behaviours C++20
//! disallows.
//!
//! Rust follows the full C++20 memory model (except for the Consume ordering). It is therefore
//! possible for this implementation to generate behaviours never observable when the same program is compiled and
//! run natively. Unfortunately, no literature exists at the time of writing which proposes an implementable and C++20-compatible
//! relaxed memory model that supports all atomic operation existing in Rust. The closest one is
//! A Promising Semantics for Relaxed-Memory Concurrency by Jeehoon Kang et al. (https://www.cs.tau.ac.il/~orilahav/papers/popl17.pdf)
//! However, this model lacks SC accesses and is therefore unusable by Miri (SC accesses are everywhere in library code).
//!
//! If you find anything that proposes a relaxed memory model that is C++20-consistent, supports all orderings Rust's atomic accesses
//! and fences accept, and is implementable (with operational semanitcs), please open a GitHub issue!
//!
//! One characteristic of this implementation, in contrast to some other notable operational models such as ones proposed in
//! Taming Release-Acquire Consistency by Ori Lahav et al. (https://plv.mpi-sws.org/sra/paper.pdf) or Promising Semantics noted above,
//! is that this implementation does not require each thread to hold an isolated view of the entire memory. Here, store buffers are per-location
//! and shared across all threads. This is more memory efficient but does require store elements (representing writes to a location) to record
//! information about reads, whereas in the other two models it is the other way round: reads points to the write it got its value from.
//! Additionally, writes in our implementation do not have globally unique timestamps attached. In the other two models this timestamp is
//! used to make sure a value in a thread's view is not overwritten by a write that occured earlier than the one in the existing view.
//! In our implementation, this is detected using read information attached to store elements, as there is no data strucutre representing reads.

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

use crate::{AtomicReadOp, AtomicRwOp, AtomicWriteOp, Tag, VClock, VTimestamp, VectorIdx};

use super::{
    allocation_map::{AccessType, AllocationMap},
    data_race::{GlobalState, ThreadClockSet},
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
    store_buffer: RefCell<AllocationMap<StoreBuffer>>,
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
    val: ScalarMaybeUninit<Tag>,

    /// Timestamp of first loads from this store element by each thread
    /// Behind a RefCell to keep load op take &self
    loads: RefCell<FxHashMap<VectorIdx, VTimestamp>>,
}

impl StoreBufferAlloc {
    pub fn new_allocation() -> Self {
        Self { store_buffer: RefCell::new(AllocationMap::new()) }
    }

    /// Gets a store buffer associated with an atomic object in this allocation
    fn get_store_buffer(&self, range: AllocRange) -> Ref<'_, StoreBuffer> {
        let access_type = self.store_buffer.borrow().access_type(range);
        let index = match access_type {
            AccessType::PerfectlyOverlapping(index) => index,
            AccessType::Empty(index) => {
                // First atomic access on this range, allocate a new StoreBuffer
                let mut buffer = self.store_buffer.borrow_mut();
                buffer.insert(index, range, StoreBuffer::default());
                index
            }
            AccessType::ImperfectlyOverlapping(index_range) => {
                // Accesses that imperfectly overlaps with existing atomic objects
                // do not have well-defined behaviours. But we don't throw a UB here
                // because we have (or will) checked that all bytes in the current
                // access are non-racy.
                // The behaviour here is that we delete all the existing objects this
                // access touches, and allocate a new and empty one for the exact range.
                // A read on an empty buffer returns None, which means the program will
                // observe the latest value in modification order at every byte.
                let mut buffer = self.store_buffer.borrow_mut();
                for index in index_range.clone() {
                    buffer.remove(index);
                }
                buffer.insert(index_range.start, range, StoreBuffer::default());
                index_range.start
            }
        };
        Ref::map(self.store_buffer.borrow(), |buffer| &buffer[index])
    }

    /// Gets a mutable store buffer associated with an atomic object in this allocation
    fn get_store_buffer_mut(&mut self, range: AllocRange) -> &mut StoreBuffer {
        let buffer = self.store_buffer.get_mut();
        let access_type = buffer.access_type(range);
        let index = match access_type {
            AccessType::PerfectlyOverlapping(index) => index,
            AccessType::Empty(index) => {
                buffer.insert(index, range, StoreBuffer::default());
                index
            }
            AccessType::ImperfectlyOverlapping(index_range) => {
                for index in index_range.clone() {
                    buffer.remove(index);
                }
                buffer.insert(index_range.start, range, StoreBuffer::default());
                index_range.start
            }
        };
        &mut buffer[index]
    }
}

impl Default for StoreBuffer {
    fn default() -> Self {
        let mut buffer = VecDeque::new();
        buffer.reserve(STORE_BUFFER_LIMIT);
        Self { buffer }
    }
}

impl<'mir, 'tcx: 'mir> StoreBuffer {
    /// Reads from the last store in modification order
    pub(super) fn read_from_last_store(&self, global: &GlobalState) {
        let store_elem = self.buffer.back();
        if let Some(store_elem) = store_elem {
            let (index, clocks) = global.current_thread_state();
            store_elem.load_impl(index, &clocks);
        }
    }

    pub(super) fn buffered_read(
        &self,
        global: &GlobalState,
        is_seqcst: bool,
        rng: &mut (impl rand::Rng + ?Sized),
        validate: impl FnOnce() -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, Option<ScalarMaybeUninit<Tag>>> {
        // Having a live borrow to store_buffer while calling validate_atomic_load is fine
        // because the race detector doesn't touch store_buffer

        let store_elem = {
            // The `clocks` we got here must be dropped before calling validate_atomic_load
            // as the race detector will update it
            let (.., clocks) = global.current_thread_state();
            // Load from a valid entry in the store buffer
            self.fetch_store(is_seqcst, &clocks, &mut *rng)
        };

        // Unlike in buffered_atomic_write, thread clock updates have to be done
        // after we've picked a store element from the store buffer, as presented
        // in ATOMIC LOAD rule of the paper. This is because fetch_store
        // requires access to ThreadClockSet.clock, which is updated by the race detector
        validate()?;

        let loaded = store_elem.map(|store_elem| {
            let (index, clocks) = global.current_thread_state();
            store_elem.load_impl(index, &clocks)
        });
        Ok(loaded)
    }

    pub(super) fn buffered_write(
        &mut self,
        val: ScalarMaybeUninit<Tag>,
        global: &GlobalState,
        is_seqcst: bool,
    ) -> InterpResult<'tcx> {
        let (index, clocks) = global.current_thread_state();

        self.store_impl(val, index, &clocks.clock, is_seqcst);
        Ok(())
    }

    /// Selects a valid store element in the buffer.
    /// The buffer does not contain the value used to initialise the atomic object
    /// so a fresh atomic object has an empty store buffer and this function
    /// will return `None`. In this case, the caller should ensure that the non-buffered
    /// value from `MiriEvalContext::read_scalar()` is observed by the program, which is
    /// the initial value of the atomic object. `MiriEvalContext::read_scalar()` is always
    /// the latest value in modification order so it is always correct to be observed by any thread.
    fn fetch_store<R: rand::Rng + ?Sized>(
        &self,
        is_seqcst: bool,
        clocks: &ThreadClockSet,
        rng: &mut R,
    ) -> Option<&StoreElement> {
        use rand::seq::IteratorRandom;
        let mut found_sc = false;
        // FIXME: this should be an inclusive take_while (stops after a false predicate, but
        // includes the element that gave the false), but such function doesn't yet
        // exist in the standard libary https://github.com/rust-lang/rust/issues/62208
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

        candidates.choose(rng)
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
    fn buffered_atomic_rmw(
        &mut self,
        new_val: ScalarMaybeUninit<Tag>,
        place: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicRwOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
        if let (
            crate::AllocExtra { weak_memory: Some(alloc_buffers), .. },
            crate::Evaluator { data_race: Some(global), .. },
        ) = this.get_alloc_extra_mut(alloc_id)?
        {
            if atomic == AtomicRwOp::SeqCst {
                global.sc_read();
                global.sc_write();
            }
            let range = alloc_range(base_offset, place.layout.size);
            let buffer = alloc_buffers.get_store_buffer_mut(range);
            buffer.read_from_last_store(global);
            buffer.buffered_write(new_val, global, atomic == AtomicRwOp::SeqCst)?;
        }
        Ok(())
    }

    fn buffered_atomic_read(
        &self,
        place: &MPlaceTy<'tcx, Tag>,
        atomic: AtomicReadOp,
        latest_in_mo: ScalarMaybeUninit<Tag>,
        validate: impl FnOnce() -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_ref();
        if let Some(global) = &this.machine.data_race {
            let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
            if let Some(alloc_buffers) = this.get_alloc_extra(alloc_id)?.weak_memory.as_ref() {
                if atomic == AtomicReadOp::SeqCst {
                    global.sc_read();
                }
                let mut rng = this.machine.rng.borrow_mut();
                let buffer =
                    alloc_buffers.get_store_buffer(alloc_range(base_offset, place.layout.size));
                let loaded = buffer.buffered_read(
                    global,
                    atomic == AtomicReadOp::SeqCst,
                    &mut *rng,
                    validate,
                )?;

                return Ok(loaded.unwrap_or(latest_in_mo));
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
        atomic: AtomicWriteOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(dest.ptr)?;
        if let (
            crate::AllocExtra { weak_memory: Some(alloc_buffers), .. },
            crate::Evaluator { data_race: Some(global), .. },
        ) = this.get_alloc_extra_mut(alloc_id)?
        {
            if atomic == AtomicWriteOp::SeqCst {
                global.sc_write();
            }
            let buffer =
                alloc_buffers.get_store_buffer_mut(alloc_range(base_offset, dest.layout.size));
            buffer.buffered_write(val, global, atomic == AtomicWriteOp::SeqCst)?;
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
        atomic: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();

        if let Some(global) = &this.machine.data_race {
            if atomic == AtomicReadOp::SeqCst {
                global.sc_read();
            }
            let size = place.layout.size;
            let (alloc_id, base_offset, ..) = this.ptr_get_alloc_id(place.ptr)?;
            if let Some(alloc_buffers) = this.get_alloc_extra(alloc_id)?.weak_memory.as_ref() {
                let buffer = alloc_buffers.get_store_buffer(alloc_range(base_offset, size));
                buffer.read_from_last_store(global);
            }
        }
        Ok(())
    }
}
