//!
use oorandom::Rand64;
use parking_lot::Mutex;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use triomphe::Arc;

/// A simple and approximate concurrent lru list.
///
/// We assume but do not verify that each node is only used with one
/// list. If this is not the case, it is not *unsafe*, but panics and
/// weird results will ensue.
///
/// Each "node" in the list is of type `Node` and must implement
/// `LruNode`, which is a trait that gives access to a field that
/// stores the index in the list. This index gives us a rough idea of
/// how recently the node has been used.
#[derive(Debug)]
pub(crate) struct Lru<Node>
where
    Node: LruNode,
{
    green_zone: AtomicUsize,
    data: Mutex<LruData<Node>>,
}

#[derive(Debug)]
struct LruData<Node> {
    end_red_zone: usize,
    end_yellow_zone: usize,
    end_green_zone: usize,
    rng: Rand64,
    entries: Vec<Arc<Node>>,
}

pub(crate) trait LruNode: Sized + Debug {
    fn lru_index(&self) -> &LruIndex;
}

#[derive(Debug)]
pub(crate) struct LruIndex {
    /// Index in the appropriate LRU list, or std::usize::MAX if not a
    /// member.
    index: AtomicUsize,
}

impl<Node> Default for Lru<Node>
where
    Node: LruNode,
{
    fn default() -> Self {
        Lru::new()
    }
}

// We always use a fixed seed for our randomness so that we have
// predictable results.
const LRU_SEED: &str = "Hello, Rustaceans";

impl<Node> Lru<Node>
where
    Node: LruNode,
{
    /// Creates a new LRU list where LRU caching is disabled.
    pub(crate) fn new() -> Self {
        Self::with_seed(LRU_SEED)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn with_seed(seed: &str) -> Self {
        Lru { green_zone: AtomicUsize::new(0), data: Mutex::new(LruData::with_seed(seed)) }
    }

    /// Adjust the total number of nodes permitted to have a value at
    /// once.  If `len` is zero, this disables LRU caching completely.
    pub(crate) fn set_lru_capacity(&self, len: usize) {
        let mut data = self.data.lock();

        // We require each zone to have at least 1 slot. Therefore,
        // the length cannot be just 1 or 2.
        if len == 0 {
            self.green_zone.store(0, Ordering::Release);
            data.resize(0, 0, 0);
        } else {
            let len = std::cmp::max(len, 3);

            // Top 10% is the green zone. This must be at least length 1.
            let green_zone = std::cmp::max(len / 10, 1);

            // Next 20% is the yellow zone.
            let yellow_zone = std::cmp::max(len / 5, 1);

            // Remaining 70% is the red zone.
            let red_zone = len - yellow_zone - green_zone;

            // We need quick access to the green zone.
            self.green_zone.store(green_zone, Ordering::Release);

            // Resize existing array.
            data.resize(green_zone, yellow_zone, red_zone);
        }
    }

    /// Records that `node` was used. This may displace an old node (if the LRU limits are
    pub(crate) fn record_use(&self, node: &Arc<Node>) -> Option<Arc<Node>> {
        tracing::debug!("record_use(node={:?})", node);

        // Load green zone length and check if the LRU cache is even enabled.
        let green_zone = self.green_zone.load(Ordering::Acquire);
        tracing::debug!("record_use: green_zone={}", green_zone);
        if green_zone == 0 {
            return None;
        }

        // Find current index of list (if any) and the current length
        // of our green zone.
        let index = node.lru_index().load();
        tracing::debug!("record_use: index={}", index);

        // Already a member of the list, and in the green zone -- nothing to do!
        if index < green_zone {
            return None;
        }

        self.data.lock().record_use(node)
    }

    pub(crate) fn purge(&self) {
        self.green_zone.store(0, Ordering::SeqCst);
        *self.data.lock() = LruData::with_seed(LRU_SEED);
    }
}

impl<Node> LruData<Node>
where
    Node: LruNode,
{
    fn with_seed(seed_str: &str) -> Self {
        Self::with_rng(rng_with_seed(seed_str))
    }

    fn with_rng(rng: Rand64) -> Self {
        LruData { end_yellow_zone: 0, end_green_zone: 0, end_red_zone: 0, entries: Vec::new(), rng }
    }

    fn green_zone(&self) -> std::ops::Range<usize> {
        0..self.end_green_zone
    }

    fn yellow_zone(&self) -> std::ops::Range<usize> {
        self.end_green_zone..self.end_yellow_zone
    }

    fn red_zone(&self) -> std::ops::Range<usize> {
        self.end_yellow_zone..self.end_red_zone
    }

    fn resize(&mut self, len_green_zone: usize, len_yellow_zone: usize, len_red_zone: usize) {
        self.end_green_zone = len_green_zone;
        self.end_yellow_zone = self.end_green_zone + len_yellow_zone;
        self.end_red_zone = self.end_yellow_zone + len_red_zone;
        let entries = std::mem::replace(&mut self.entries, Vec::with_capacity(self.end_red_zone));

        tracing::debug!("green_zone = {:?}", self.green_zone());
        tracing::debug!("yellow_zone = {:?}", self.yellow_zone());
        tracing::debug!("red_zone = {:?}", self.red_zone());

        // We expect to resize when the LRU cache is basically empty.
        // So just forget all the old LRU indices to start.
        for entry in entries {
            entry.lru_index().clear();
        }
    }

    /// Records that a node was used. If it is already a member of the
    /// LRU list, it is promoted to the green zone (unless it's
    /// already there). Otherwise, it is added to the list first and
    /// *then* promoted to the green zone. Adding a new node to the
    /// list may displace an old member of the red zone, in which case
    /// that is returned.
    fn record_use(&mut self, node: &Arc<Node>) -> Option<Arc<Node>> {
        tracing::debug!("record_use(node={:?})", node);

        // NB: When this is invoked, we have typically already loaded
        // the LRU index (to check if it is in green zone). But that
        // check was done outside the lock and -- for all we know --
        // the index may have changed since. So we always reload.
        let index = node.lru_index().load();

        if index < self.end_green_zone {
            None
        } else if index < self.end_yellow_zone {
            self.promote_yellow_to_green(node, index);
            None
        } else if index < self.end_red_zone {
            self.promote_red_to_green(node, index);
            None
        } else {
            self.insert_new(node)
        }
    }

    /// Inserts a node that is not yet a member of the LRU list. If
    /// the list is at capacity, this can displace an existing member.
    fn insert_new(&mut self, node: &Arc<Node>) -> Option<Arc<Node>> {
        debug_assert!(!node.lru_index().is_in_lru());

        // Easy case: we still have capacity. Push it, and then promote
        // it up to the appropriate zone.
        let len = self.entries.len();
        if len < self.end_red_zone {
            self.entries.push(node.clone());
            node.lru_index().store(len);
            tracing::debug!("inserted node {:?} at {}", node, len);
            return self.record_use(node);
        }

        // Harder case: no capacity. Create some by evicting somebody from red
        // zone and then promoting.
        let victim_index = self.pick_index(self.red_zone());
        let victim_node = std::mem::replace(&mut self.entries[victim_index], node.clone());
        tracing::debug!("evicting red node {:?} from {}", victim_node, victim_index);
        victim_node.lru_index().clear();
        self.promote_red_to_green(node, victim_index);
        Some(victim_node)
    }

    /// Promotes the node `node`, stored at `red_index` (in the red
    /// zone), into a green index, demoting yellow/green nodes at
    /// random.
    ///
    /// NB: It is not required that `node.lru_index()` is up-to-date
    /// when entering this method.
    fn promote_red_to_green(&mut self, node: &Arc<Node>, red_index: usize) {
        debug_assert!(self.red_zone().contains(&red_index));

        // Pick a yellow at random and switch places with it.
        //
        // Subtle: we do not update `node.lru_index` *yet* -- we're
        // going to invoke `self.promote_yellow` next, and it will get
        // updated then.
        let yellow_index = self.pick_index(self.yellow_zone());
        tracing::debug!(
            "demoting yellow node {:?} from {} to red at {}",
            self.entries[yellow_index],
            yellow_index,
            red_index,
        );
        self.entries.swap(yellow_index, red_index);
        self.entries[red_index].lru_index().store(red_index);

        // Now move ourselves up into the green zone.
        self.promote_yellow_to_green(node, yellow_index);
    }

    /// Promotes the node `node`, stored at `yellow_index` (in the
    /// yellow zone), into a green index, demoting a green node at
    /// random to replace it.
    ///
    /// NB: It is not required that `node.lru_index()` is up-to-date
    /// when entering this method.
    fn promote_yellow_to_green(&mut self, node: &Arc<Node>, yellow_index: usize) {
        debug_assert!(self.yellow_zone().contains(&yellow_index));

        // Pick a yellow at random and switch places with it.
        let green_index = self.pick_index(self.green_zone());
        tracing::debug!(
            "demoting green node {:?} from {} to yellow at {}",
            self.entries[green_index],
            green_index,
            yellow_index
        );
        self.entries.swap(green_index, yellow_index);
        self.entries[yellow_index].lru_index().store(yellow_index);
        node.lru_index().store(green_index);

        tracing::debug!("promoted {:?} to green index {}", node, green_index);
    }

    fn pick_index(&mut self, zone: std::ops::Range<usize>) -> usize {
        let end_index = std::cmp::min(zone.end, self.entries.len());
        self.rng.rand_range(zone.start as u64..end_index as u64) as usize
    }
}

impl Default for LruIndex {
    fn default() -> Self {
        Self { index: AtomicUsize::new(std::usize::MAX) }
    }
}

impl LruIndex {
    fn load(&self) -> usize {
        self.index.load(Ordering::Acquire) // see note on ordering below
    }

    fn store(&self, value: usize) {
        self.index.store(value, Ordering::Release) // see note on ordering below
    }

    fn clear(&self) {
        self.store(std::usize::MAX);
    }

    fn is_in_lru(&self) -> bool {
        self.load() != std::usize::MAX
    }
}

fn rng_with_seed(seed_str: &str) -> Rand64 {
    let mut seed: [u8; 16] = [0; 16];
    for (i, &b) in seed_str.as_bytes().iter().take(16).enumerate() {
        seed[i] = b;
    }
    Rand64::new(u128::from_le_bytes(seed))
}

// A note on ordering:
//
// I chose to use AcqRel for the ordering but I don't think it's
// strictly needed.  All writes occur under a lock, so they should be
// ordered w/r/t one another.  As for the reads, they can occur
// outside the lock, but they don't themselves enable dependent reads
// -- if the reads are out of bounds, we would acquire a lock.
