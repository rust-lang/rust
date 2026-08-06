//! How a task's reads are recorded and deduplicated into the edge list of its node.

use rustc_data_structures::sync::Lock;
use rustc_index::IndexVec;

use super::DepNodeIndex;

/// How many reads fit in [`TaskReads::Small`]'s inline buffer.
pub(crate) const SMALL_READS_MAX: usize = 16;

/// The reads recorded by one task so far, deduplicated and in first-read order.
#[derive(Debug)]
pub(crate) enum TaskReads {
    /// The first few reads, deduplicated by a linear scan. Most tasks never outgrow this.
    Small { len: u8, buf: [DepNodeIndex; SMALL_READS_MAX] },

    /// The reads of a task that outgrew the inline buffer.
    Recorded(ReadsRecorder),
}

impl TaskReads {
    #[inline]
    pub(crate) fn new() -> Self {
        TaskReads::Small { len: 0, buf: [DepNodeIndex::ZERO; SMALL_READS_MAX] }
    }

    /// Records a read and returns whether it was new for the task. The pool provides a
    /// recorder when the task outgrows the inline buffer.
    #[inline]
    pub(crate) fn insert(&mut self, index: DepNodeIndex, pool: &Lock<Vec<ReadsRecorder>>) -> bool {
        match self {
            TaskReads::Recorded(recorder) => recorder.insert(index),
            TaskReads::Small { len, buf } => {
                let n = usize::from(*len);
                if buf[..n].contains(&index) {
                    false
                } else if n < SMALL_READS_MAX {
                    buf[n] = index;
                    *len += 1;
                    true
                } else {
                    // The inline buffer is full: move the reads into a pooled recorder.
                    let seed = *buf;
                    let mut recorder = pool.lock().pop().unwrap_or_default();
                    recorder.clear();
                    recorder.seed(&seed);
                    recorder.insert(index);
                    *self = TaskReads::Recorded(recorder);
                    true
                }
            }
        }
    }

    /// The task's deduplicated reads, in first-read order.
    #[inline]
    pub(crate) fn edges(&self) -> &[DepNodeIndex] {
        match self {
            TaskReads::Small { len, buf } => &buf[..usize::from(*len)],
            TaskReads::Recorded(recorder) => &recorder.reads,
        }
    }
}

/// Records the reads of a task with many reads.
///
/// Recorders are pooled globally, reusing the read list and epoch filter allocations
/// across tasks.
#[derive(Debug, Default)]
pub(crate) struct ReadsRecorder {
    /// The deduplicated reads, in first-read order.
    reads: Vec<DepNodeIndex>,
    /// Seen-before filter: a slot counts as seen only while it holds `epoch`.
    epochs: IndexVec<DepNodeIndex, u8>,
    epoch: u8,
}

impl ReadsRecorder {
    /// Seeds a fresh recorder with reads that are already known to be distinct.
    fn seed(&mut self, reads: &[DepNodeIndex]) {
        debug_assert!(self.reads.is_empty());
        let epoch = self.epoch;
        for &index in reads {
            *self.slot(index) = epoch;
        }
        self.reads.extend_from_slice(reads);
    }

    /// Records a read of `index` and returns whether it was new for the current task.
    #[inline]
    fn insert(&mut self, index: DepNodeIndex) -> bool {
        let epoch = self.epoch;
        let slot = self.slot(index);
        if *slot == epoch {
            false
        } else {
            *slot = epoch;
            self.reads.push(index);
            true
        }
    }

    /// The epoch slot for `index`, growing the filter (with slack) to cover it.
    #[inline]
    fn slot(&mut self, index: DepNodeIndex) -> &mut u8 {
        let index = index.as_usize();
        if index >= self.epochs.raw.len() {
            // New indices keep showing up all session, so growing to exactly
            // `index + 1` (e.g. `ensure_contains_elem`) would resize again and
            // again. The power-of-two slack makes resizes rare, and starting
            // at 64 skips the smallest sizes.
            self.epochs.raw.resize((index + 1).next_power_of_two().max(64), 0);
        }
        &mut self.epochs.raw[index]
    }

    /// Prepares the recorder for a new task, keeping the backing allocations. Bumping the
    /// epoch invalidates every slot at once; on wraparound the slots are zeroed.
    fn clear(&mut self) {
        self.reads.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epochs.raw.fill(0);
            self.epoch = 1;
        }
    }

    /// Returns a finished task's recorder to the pool. The cap keeps deeply nested tasks
    /// from growing the pool without bound.
    #[inline]
    pub(crate) fn release(self, pool: &Lock<Vec<ReadsRecorder>>) {
        const MAX_POOLED_RECORDERS: usize = 4;
        let mut pool = pool.lock();
        if pool.len() < MAX_POOLED_RECORDERS {
            pool.push(self);
        }
    }
}
