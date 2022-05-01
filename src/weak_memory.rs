//! Implementation of C++11-consistent weak memory emulation using store buffers
//! based on Dynamic Race Detection for C++ ("the paper"):
//! https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2017/POPL.pdf

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
    cell::{Ref, RefCell, RefMut},
    collections::VecDeque,
};

use rustc_const_eval::interpret::{AllocRange, InterpResult, ScalarMaybeUninit};
use rustc_data_structures::fx::FxHashMap;
use rustc_target::abi::Size;

use crate::{
    data_race::{GlobalState, ThreadClockSet},
    RangeMap, Tag, VClock, VTimestamp, VectorIdx,
};

pub type AllocExtra = StoreBufferAlloc;
#[derive(Debug, Clone)]
pub struct StoreBufferAlloc {
    /// Store buffer of each atomic object in this allocation
    // Load may modify a StoreBuffer to record the loading thread's
    // timestamp so we need interior mutability here.
    store_buffer: RefCell<RangeMap<StoreBuffer>>,
}

impl StoreBufferAlloc {
    pub fn new_allocation(len: Size) -> Self {
        Self { store_buffer: RefCell::new(RangeMap::new(len, StoreBuffer::default())) }
    }

    /// Gets a store buffer associated with an atomic object in this allocation
    pub fn get_store_buffer(&self, range: AllocRange) -> Ref<'_, StoreBuffer> {
        Ref::map(self.store_buffer.borrow(), |range_map| {
            let (.., store_buffer) = range_map.iter(range.start, range.size).next().unwrap();
            store_buffer
        })
    }

    pub fn get_store_buffer_mut(&self, range: AllocRange) -> RefMut<'_, StoreBuffer> {
        RefMut::map(self.store_buffer.borrow_mut(), |range_map| {
            let (.., store_buffer) = range_map.iter_mut(range.start, range.size).next().unwrap();
            store_buffer
        })
    }

}

const STORE_BUFFER_LIMIT: usize = 128;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreBuffer {
    // Stores to this location in modification order
    buffer: VecDeque<StoreElement>,
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
    pub fn read_from_last_store(&self, global: &GlobalState) {
        let store_elem = self.buffer.back();
        if let Some(store_elem) = store_elem {
            let (index, clocks) = global.current_thread_state();
            store_elem.load_impl(index, &clocks);
        }
    }

    pub fn buffered_read(
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

        // Unlike in write_scalar_atomic, thread clock updates have to be done
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

    pub fn buffered_write(
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
    /// so a fresh atomic object has an empty store buffer until an explicit store.
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
                // CoWR: if a store happens-before the current load,
                // then we can't read-from anything earlier in modification order.
                if store_elem.timestamp <= clocks.clock[store_elem.store_index] {
                    log::info!("Stopped due to coherent write-read");
                    keep_searching = false;
                    return true;
                }

                // CoRR: if there was a load from this store which happened-before the current load,
                // then we cannot read-from anything earlier in modification order.
                if store_elem.loads.borrow().iter().any(|(&load_index, &load_timestamp)| {
                    load_timestamp <= clocks.clock[load_index]
                }) {
                    log::info!("Stopped due to coherent read-read");
                    keep_searching = false;
                    return true;
                }

                // The current load, which may be sequenced-after an SC fence, can only read-from
                // the last store sequenced-before an SC fence in another thread (or any stores
                // later than that SC fence)
                if store_elem.timestamp <= clocks.fence_seqcst[store_elem.store_index] {
                    log::info!("Stopped due to coherent load sequenced after sc fence");
                    keep_searching = false;
                    return true;
                }

                // The current non-SC load can only read-from the latest SC store (or any stores later than that
                // SC store)
                if store_elem.timestamp <= clocks.write_seqcst[store_elem.store_index]
                    && store_elem.is_seqcst
                {
                    log::info!("Stopped due to needing to load from the last SC store");
                    keep_searching = false;
                    return true;
                }

                // The current SC load can only read-from the last store sequenced-before
                // the last SC fence (or any stores later than the SC fence)
                if is_seqcst && store_elem.timestamp <= clocks.read_seqcst[store_elem.store_index] {
                    log::info!("Stopped due to sc load needing to load from the last SC store before an SC fence");
                    keep_searching = false;
                    return true;
                }

                true
            })
            .filter(|&store_elem| {
                if is_seqcst {
                    // An SC load needs to ignore all but last store maked SC (stores not marked SC are not
                    // affected)
                    let include = !(store_elem.is_seqcst && found_sc);
                    found_sc |= store_elem.is_seqcst;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreElement {
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
