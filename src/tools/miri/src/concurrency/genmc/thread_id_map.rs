use genmc_sys::GENMC_MAIN_THREAD_ID;
use rustc_data_structures::fx::FxHashMap;

use crate::ThreadId;

#[derive(Debug)]
pub struct ThreadIdMap {
    /// Map from Miri thread IDs to GenMC thread IDs.
    /// We assume as little as possible about Miri thread IDs, so we use a map.
    miri_to_genmc: FxHashMap<ThreadId, i32>,
    /// Map from GenMC thread IDs to Miri thread IDs.
    /// We control which thread IDs are used, so we choose them in as an incrementing counter.
    genmc_to_miri: Vec<ThreadId>, // FIXME(genmc): check if this assumption is (and will stay) correct.
}

impl Default for ThreadIdMap {
    fn default() -> Self {
        let miri_to_genmc = [(ThreadId::MAIN_THREAD, GENMC_MAIN_THREAD_ID)].into_iter().collect();
        let genmc_to_miri = vec![ThreadId::MAIN_THREAD];
        Self { miri_to_genmc, genmc_to_miri }
    }
}

impl ThreadIdMap {
    pub fn reset(&mut self) {
        self.miri_to_genmc.clear();
        self.miri_to_genmc.insert(ThreadId::MAIN_THREAD, GENMC_MAIN_THREAD_ID);
        self.genmc_to_miri.clear();
        self.genmc_to_miri.push(ThreadId::MAIN_THREAD);
    }

    #[must_use]
    /// Add a new Miri thread to the mapping and dispense a new thread ID for GenMC to use.
    pub fn add_thread(&mut self, thread_id: ThreadId) -> i32 {
        // NOTE: We select the new thread ids as integers incremented by one (we use the length as the counter).
        let next_thread_id = self.genmc_to_miri.len();
        let genmc_tid = next_thread_id.try_into().unwrap();
        // If there is already an entry, we override it.
        // This could happen if Miri were to reuse `ThreadId`s, but we assume that if this happens, the previous thread with that id doesn't exist anymore.
        self.miri_to_genmc.insert(thread_id, genmc_tid);
        self.genmc_to_miri.push(thread_id);

        genmc_tid
    }

    #[must_use]
    /// Try to get the GenMC thread ID corresponding to a given Miri `ThreadId`.
    /// Panics if there is no mapping for the given `ThreadId`.
    pub fn get_genmc_tid(&self, thread_id: ThreadId) -> i32 {
        *self.miri_to_genmc.get(&thread_id).unwrap()
    }

    #[must_use]
    /// Get the Miri `ThreadId` corresponding to a given GenMC thread id.
    /// Panics if the given thread id isn't valid.
    pub fn get_miri_tid(&self, genmc_tid: i32) -> ThreadId {
        let index: usize = genmc_tid.try_into().unwrap();
        self.genmc_to_miri
            .get(index)
            .copied()
            .expect("A thread id returned from GenMC should exist.")
    }
}
