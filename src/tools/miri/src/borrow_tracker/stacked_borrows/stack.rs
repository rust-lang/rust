#[cfg(feature = "stack-cache")]
use std::ops::Range;

use rustc_data_structures::fx::FxHashSet;
use tracing::trace;

use crate::borrow_tracker::stacked_borrows::{Item, Permission};
use crate::borrow_tracker::{AccessKind, BorTag};
use crate::{InterpResult, ProvenanceExtra, interp_ok};

/// Exactly what cache size we should use is a difficult trade-off. There will always be some
/// workload which has a `BorTag` working set which exceeds the size of the cache, and ends up
/// falling back to linear searches of the borrow stack very often.
/// The cost of making this value too large is that the loop in `Stack::insert` which ensures the
/// entries in the cache stay correct after an insert becomes expensive.
#[cfg(feature = "stack-cache")]
const CACHE_LEN: usize = 32;

/// Extra per-location state.
#[derive(Clone, Debug)]
pub struct Stack {
    /// Used *mostly* as a stack; never empty.
    /// Invariants:
    /// * Above a `SharedReadOnly` there can only be more `SharedReadOnly`.
    /// * Except for `Untagged`, no tag occurs in the stack more than once.
    borrows: Vec<Item>,
    /// If this is `Some(id)`, then the actual current stack is unknown. This can happen when
    /// wildcard pointers are used to access this location. What we do know is that `borrows` are at
    /// the top of the stack, and below it are arbitrarily many items whose `tag` is strictly less
    /// than `id`.
    /// When the bottom is unknown, `borrows` always has a `SharedReadOnly` or `Unique` at the bottom;
    /// we never have the unknown-to-known boundary in an SRW group.
    unknown_bottom: Option<BorTag>,

    /// A small LRU cache of searches of the borrow stack.
    #[cfg(feature = "stack-cache")]
    cache: StackCache,
    /// On a read, we need to disable all `Unique` above the granting item. We can avoid most of
    /// this scan by keeping track of the region of the borrow stack that may contain `Unique`s.
    #[cfg(feature = "stack-cache")]
    unique_range: Range<usize>,
}

impl Stack {
    pub fn retain(&mut self, tags: &FxHashSet<BorTag>) {
        let mut first_removed = None;

        // We never consider removing the bottom-most tag. For stacks without an unknown
        // bottom this preserves the root tag.
        // Note that the algorithm below is based on considering the tag at read_idx - 1,
        // so precisely considering the tag at index 0 for removal when we have an unknown
        // bottom would complicate the implementation. The simplification of not considering
        // it does not have a significant impact on the degree to which the GC mitigates
        // memory growth.
        let mut read_idx = 1;
        let mut write_idx = read_idx;
        while read_idx < self.borrows.len() {
            let left = self.borrows[read_idx - 1];
            let this = self.borrows[read_idx];
            let should_keep = match this.perm() {
                // SharedReadWrite is the simplest case, if it's unreachable we can just remove it.
                Permission::SharedReadWrite => tags.contains(&this.tag()),
                // Only retain a Disabled tag if it is terminating a SharedReadWrite block.
                Permission::Disabled => left.perm() == Permission::SharedReadWrite,
                // Unique and SharedReadOnly can terminate a SharedReadWrite block, so only remove
                // them if they are both unreachable and not directly after a SharedReadWrite.
                Permission::Unique | Permission::SharedReadOnly =>
                    left.perm() == Permission::SharedReadWrite || tags.contains(&this.tag()),
            };

            if should_keep {
                if read_idx != write_idx {
                    self.borrows[write_idx] = self.borrows[read_idx];
                }
                write_idx += 1;
            } else if first_removed.is_none() {
                first_removed = Some(read_idx);
            }

            read_idx += 1;
        }
        self.borrows.truncate(write_idx);

        #[cfg(not(feature = "stack-cache"))]
        let _unused = first_removed; // This is only needed for the stack-cache

        #[cfg(feature = "stack-cache")]
        if let Some(first_removed) = first_removed {
            // Either end of unique_range may have shifted, all we really know is that we can't
            // have introduced a new Unique.
            if !self.unique_range.is_empty() {
                self.unique_range = 0..self.len();
            }

            // Replace any Items which have been collected with the root item, a known-good value.
            for i in 0..CACHE_LEN {
                if self.cache.idx[i] >= first_removed {
                    self.cache.items[i] = self.borrows[0];
                    self.cache.idx[i] = 0;
                }
            }
        }
    }
}

/// A very small cache of searches of a borrow stack, mapping `Item`s to their position in said stack.
///
/// It may seem like maintaining this cache is a waste for small stacks, but
/// (a) iterating over small fixed-size arrays is super fast, and (b) empirically this helps *a lot*,
/// probably because runtime is dominated by large stacks.
#[cfg(feature = "stack-cache")]
#[derive(Clone, Debug)]
struct StackCache {
    items: [Item; CACHE_LEN], // Hot in find_granting
    idx: [usize; CACHE_LEN],  // Hot in grant
}

#[cfg(feature = "stack-cache")]
impl StackCache {
    /// When a tag is used, we call this function to add or refresh it in the cache.
    ///
    /// We use the position in the cache to represent how recently a tag was used; the first position
    /// is the most recently used tag. So an add shifts every element towards the end, and inserts
    /// the new element at the start. We lose the last element.
    /// This strategy is effective at keeping the most-accessed items in the cache, but it costs a
    /// linear shift across the entire cache when we add a new tag.
    fn add(&mut self, idx: usize, item: Item) {
        self.items.copy_within(0..CACHE_LEN - 1, 1);
        self.items[0] = item;
        self.idx.copy_within(0..CACHE_LEN - 1, 1);
        self.idx[0] = idx;
    }
}

impl PartialEq for Stack {
    fn eq(&self, other: &Self) -> bool {
        let Stack {
            borrows,
            unknown_bottom,
            // The cache is ignored for comparison.
            #[cfg(feature = "stack-cache")]
                cache: _,
            #[cfg(feature = "stack-cache")]
                unique_range: _,
        } = self;
        *borrows == other.borrows && *unknown_bottom == other.unknown_bottom
    }
}

impl Eq for Stack {}

impl<'tcx> Stack {
    /// Panics if any of the caching mechanisms have broken,
    /// - The StackCache indices don't refer to the parallel items,
    /// - There are no Unique items outside of first_unique..last_unique
    #[cfg(feature = "stack-cache-consistency-check")]
    fn verify_cache_consistency(&self) {
        // Only a full cache needs to be valid. Also see the comments in find_granting_cache
        // and set_unknown_bottom.
        if self.borrows.len() >= CACHE_LEN {
            for (tag, stack_idx) in self.cache.items.iter().zip(self.cache.idx.iter()) {
                assert_eq!(self.borrows[*stack_idx], *tag);
            }
        }

        // Check that all Unique items fall within unique_range.
        for (idx, item) in self.borrows.iter().enumerate() {
            if item.perm() == Permission::Unique {
                assert!(
                    self.unique_range.contains(&idx),
                    "{:?} {:?}",
                    self.unique_range,
                    self.borrows
                );
            }
        }

        // Check that the unique_range is a valid index into the borrow stack.
        // This asserts that the unique_range's start <= end.
        let _uniques = &self.borrows[self.unique_range.clone()];

        // We cannot assert that the unique range is precise.
        // Both ends may shift around when `Stack::retain` is called. Additionally,
        // when we pop items within the unique range, setting the end of the range precisely
        // requires doing a linear search of the borrow stack, which is exactly the kind of
        // operation that all this caching exists to avoid.
    }

    /// Find the item granting the given kind of access to the given tag, and return where
    /// it is on the stack. For wildcard tags, the given index is approximate, but if *no*
    /// index is given it means the match was *not* in the known part of the stack.
    /// `Ok(None)` indicates it matched the "unknown" part of the stack.
    /// `Err` indicates it was not found.
    pub(super) fn find_granting(
        &mut self,
        access: AccessKind,
        tag: ProvenanceExtra,
        exposed_tags: &FxHashSet<BorTag>,
    ) -> Result<Option<usize>, ()> {
        #[cfg(feature = "stack-cache-consistency-check")]
        self.verify_cache_consistency();

        let ProvenanceExtra::Concrete(tag) = tag else {
            // Handle the wildcard case.
            // Go search the stack for an exposed tag.
            if let Some(idx) = self
                .borrows
                .iter()
                .enumerate() // we also need to know *where* in the stack
                .rev() // search top-to-bottom
                .find_map(|(idx, item)| {
                    // If the item fits and *might* be this wildcard, use it.
                    if item.perm().grants(access) && exposed_tags.contains(&item.tag()) {
                        Some(idx)
                    } else {
                        None
                    }
                })
            {
                return Ok(Some(idx));
            }
            // If we couldn't find it in the stack, check the unknown bottom.
            return if self.unknown_bottom.is_some() { Ok(None) } else { Err(()) };
        };

        if let Some(idx) = self.find_granting_tagged(access, tag) {
            return Ok(Some(idx));
        }

        // Couldn't find it in the stack; but if there is an unknown bottom it might be there.
        let found = self.unknown_bottom.is_some_and(|unknown_limit| {
            tag < unknown_limit // unknown_limit is an upper bound for what can be in the unknown bottom.
        });
        if found { Ok(None) } else { Err(()) }
    }

    fn find_granting_tagged(&mut self, access: AccessKind, tag: BorTag) -> Option<usize> {
        #[cfg(feature = "stack-cache")]
        if let Some(idx) = self.find_granting_cache(access, tag) {
            return Some(idx);
        }

        // If we didn't find the tag in the cache, fall back to a linear search of the
        // whole stack, and add the tag to the cache.
        for (stack_idx, item) in self.borrows.iter().enumerate().rev() {
            if tag == item.tag() && item.perm().grants(access) {
                #[cfg(feature = "stack-cache")]
                self.cache.add(stack_idx, *item);
                return Some(stack_idx);
            }
        }
        None
    }

    #[cfg(feature = "stack-cache")]
    fn find_granting_cache(&mut self, access: AccessKind, tag: BorTag) -> Option<usize> {
        // This looks like a common-sense optimization; we're going to do a linear search of the
        // cache or the borrow stack to scan the shorter of the two. This optimization is minuscule
        // and this check actually ensures we do not access an invalid cache.
        // When a stack is created and when items are removed from the top of the borrow stack, we
        // need some valid value to populate the cache. In both cases, we try to use the bottom
        // item. But when the stack is cleared in `set_unknown_bottom` there is nothing we could
        // place in the cache that is correct. But due to the way we populate the cache in
        // `StackCache::add`, we know that when the borrow stack has grown larger than the cache,
        // every slot in the cache is valid.
        if self.borrows.len() <= CACHE_LEN {
            return None;
        }
        // Search the cache for the tag we're looking up
        let cache_idx = self.cache.items.iter().position(|t| t.tag() == tag)?;
        let stack_idx = self.cache.idx[cache_idx];
        // If we found the tag, look up its position in the stack to see if it grants
        // the required permission
        if self.cache.items[cache_idx].perm().grants(access) {
            // If it does, and it's not already in the most-recently-used position, re-insert it at
            // the most-recently-used position. This technically reduces the efficiency of the
            // cache by duplicating elements, but current benchmarks do not seem to benefit from
            // avoiding this duplication.
            // But if the tag is in position 1, avoiding the duplicating add is trivial.
            // If it does, and it's not already in the most-recently-used position, move it there.
            // Except if the tag is in position 1, this is equivalent to just a swap, so do that.
            if cache_idx == 1 {
                self.cache.items.swap(0, 1);
                self.cache.idx.swap(0, 1);
            } else if cache_idx > 1 {
                self.cache.add(stack_idx, self.cache.items[cache_idx]);
            }
            Some(stack_idx)
        } else {
            // Tag is in the cache, but it doesn't grant the required permission
            None
        }
    }

    pub fn insert(&mut self, new_idx: usize, new: Item) {
        self.borrows.insert(new_idx, new);

        #[cfg(feature = "stack-cache")]
        self.insert_cache(new_idx, new);
    }

    #[cfg(feature = "stack-cache")]
    fn insert_cache(&mut self, new_idx: usize, new: Item) {
        // Adjust the possibly-unique range if an insert occurs before or within it
        if self.unique_range.start >= new_idx {
            self.unique_range.start += 1;
        }
        if self.unique_range.end >= new_idx {
            self.unique_range.end += 1;
        }
        if new.perm() == Permission::Unique {
            // If this is the only Unique, set the range to contain just the new item.
            if self.unique_range.is_empty() {
                self.unique_range = new_idx..new_idx + 1;
            } else {
                // We already have other Unique items, expand the range to include the new item
                self.unique_range.start = self.unique_range.start.min(new_idx);
                self.unique_range.end = self.unique_range.end.max(new_idx + 1);
            }
        }

        // The above insert changes the meaning of every index in the cache >= new_idx, so now
        // we need to find every one of those indexes and increment it.
        // But if the insert is at the end (equivalent to a push), we can skip this step because
        // it didn't change the position of any other items.
        if new_idx != self.borrows.len() - 1 {
            for idx in &mut self.cache.idx {
                if *idx >= new_idx {
                    *idx += 1;
                }
            }
        }

        // This primes the cache for the next access, which is almost always the just-added tag.
        self.cache.add(new_idx, new);

        #[cfg(feature = "stack-cache-consistency-check")]
        self.verify_cache_consistency();
    }

    /// Construct a new `Stack` using the passed `Item` as the root tag.
    pub fn new(item: Item) -> Self {
        Stack {
            borrows: vec![item],
            unknown_bottom: None,
            #[cfg(feature = "stack-cache")]
            cache: StackCache { idx: [0; CACHE_LEN], items: [item; CACHE_LEN] },
            #[cfg(feature = "stack-cache")]
            unique_range: if item.perm() == Permission::Unique { 0..1 } else { 0..0 },
        }
    }

    pub fn get(&self, idx: usize) -> Option<Item> {
        self.borrows.get(idx).cloned()
    }

    #[expect(clippy::len_without_is_empty)] // Stacks are never empty
    pub fn len(&self) -> usize {
        self.borrows.len()
    }

    pub fn unknown_bottom(&self) -> Option<BorTag> {
        self.unknown_bottom
    }

    pub fn set_unknown_bottom(&mut self, tag: BorTag) {
        // We clear the borrow stack but the lookup cache doesn't support clearing per se. Instead,
        // there is a check explained in `find_granting_cache` which protects against accessing the
        // cache when it has been cleared and not yet refilled.
        self.borrows.clear();
        self.unknown_bottom = Some(tag);
        #[cfg(feature = "stack-cache")]
        {
            self.unique_range = 0..0;
        }
    }

    /// Find all `Unique` elements in this borrow stack above `granting_idx`, pass a copy of them
    /// to the `visitor`, then set their `Permission` to `Disabled`.
    pub fn disable_uniques_starting_at(
        &mut self,
        disable_start: usize,
        mut visitor: impl FnMut(Item) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        #[cfg(feature = "stack-cache")]
        let unique_range = self.unique_range.clone();
        #[cfg(not(feature = "stack-cache"))]
        let unique_range = 0..self.len();

        if disable_start <= unique_range.end {
            let lower = unique_range.start.max(disable_start);
            let upper = unique_range.end;
            for item in &mut self.borrows[lower..upper] {
                if item.perm() == Permission::Unique {
                    trace!("access: disabling item {:?}", item);
                    visitor(*item)?;
                    item.set_permission(Permission::Disabled);
                    // Also update all copies of this item in the cache.
                    #[cfg(feature = "stack-cache")]
                    for it in &mut self.cache.items {
                        if it.tag() == item.tag() {
                            it.set_permission(Permission::Disabled);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "stack-cache")]
        if disable_start <= self.unique_range.start {
            // We disabled all Unique items
            self.unique_range.start = 0;
            self.unique_range.end = 0;
        } else {
            // Truncate the range to only include items up to the index that we started disabling
            // at.
            self.unique_range.end = self.unique_range.end.min(disable_start);
        }

        #[cfg(feature = "stack-cache-consistency-check")]
        self.verify_cache_consistency();

        interp_ok(())
    }

    /// Produces an iterator which iterates over `range` in reverse, and when dropped removes that
    /// range of `Item`s from this `Stack`.
    pub fn pop_items_after<V: FnMut(Item) -> InterpResult<'tcx>>(
        &mut self,
        start: usize,
        mut visitor: V,
    ) -> InterpResult<'tcx> {
        while self.borrows.len() > start {
            let item = self.borrows.pop().unwrap();
            visitor(item)?;
        }

        #[cfg(feature = "stack-cache")]
        if !self.borrows.is_empty() {
            // After we remove from the borrow stack, every aspect of our caching may be invalid, but it is
            // also possible that the whole cache is still valid. So we call this method to repair what
            // aspects of the cache are now invalid, instead of resetting the whole thing to a trivially
            // valid default state.
            let base_tag = self.borrows[0];
            let mut removed = 0;
            let mut cursor = 0;
            // Remove invalid entries from the cache by rotating them to the end of the cache, then
            // keep track of how many invalid elements there are and overwrite them with the root tag.
            // The root tag here serves as a harmless default value.
            for _ in 0..CACHE_LEN - 1 {
                if self.cache.idx[cursor] >= start {
                    self.cache.idx[cursor..CACHE_LEN - removed].rotate_left(1);
                    self.cache.items[cursor..CACHE_LEN - removed].rotate_left(1);
                    removed += 1;
                } else {
                    cursor += 1;
                }
            }
            for i in CACHE_LEN - removed - 1..CACHE_LEN {
                self.cache.idx[i] = 0;
                self.cache.items[i] = base_tag;
            }

            if start <= self.unique_range.start {
                // We removed all the Unique items
                self.unique_range = 0..0;
            } else {
                // Ensure the range doesn't extend past the new top of the stack
                self.unique_range.end = self.unique_range.end.min(start);
            }
        } else {
            self.unique_range = 0..0;
        }

        #[cfg(feature = "stack-cache-consistency-check")]
        self.verify_cache_consistency();
        interp_ok(())
    }
}
