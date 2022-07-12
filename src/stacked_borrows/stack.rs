use crate::stacked_borrows::{AccessKind, Item, Permission, SbTag, SbTagExtra};
use rustc_data_structures::fx::FxHashSet;
#[cfg(feature = "stack-cache")]
use std::ops::Range;

/// Exactly what cache size we should use is a difficult tradeoff. There will always be some
/// workload which has a `SbTag` working set which exceeds the size of the cache, and ends up
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
    unknown_bottom: Option<SbTag>,

    /// A small LRU cache of searches of the borrow stack.
    #[cfg(feature = "stack-cache")]
    cache: StackCache,
    /// On a read, we need to disable all `Unique` above the granting item. We can avoid most of
    /// this scan by keeping track of the region of the borrow stack that may contain `Unique`s.
    #[cfg(feature = "stack-cache")]
    unique_range: Range<usize>,
}

/// A very small cache of searches of the borrow stack
/// This maps items to locations in the borrow stack. Any use of this still needs to do a
/// probably-cold random access into the borrow stack to figure out what `Permission` an
/// `SbTag` grants. We could avoid this by also storing the `Permission` in the cache, but
/// most lookups into the cache are immediately followed by access of the full borrow stack anyway.
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
        // All the semantics of Stack are in self.borrows, everything else is caching
        self.borrows == other.borrows
    }
}

impl Eq for Stack {}

impl<'tcx> Stack {
    /// Panics if any of the caching mechanisms have broken,
    /// - The StackCache indices don't refer to the parallel items,
    /// - There are no Unique items outside of first_unique..last_unique
    #[cfg(feature = "expensive-debug-assertions")]
    fn verify_cache_consistency(&self) {
        // Only a full cache needs to be valid. Also see the comments in find_granting_cache
        // and set_unknown_bottom.
        if self.borrows.len() >= CACHE_LEN {
            for (tag, stack_idx) in self.cache.items.iter().zip(self.cache.idx.iter()) {
                assert_eq!(self.borrows[*stack_idx], *tag);
            }
        }

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
    }

    /// Find the item granting the given kind of access to the given tag, and return where
    /// it is on the stack. For wildcard tags, the given index is approximate, but if *no*
    /// index is given it means the match was *not* in the known part of the stack.
    /// `Ok(None)` indicates it matched the "unknown" part of the stack.
    /// `Err` indicates it was not found.
    pub(super) fn find_granting(
        &mut self,
        access: AccessKind,
        tag: SbTagExtra,
        exposed_tags: &FxHashSet<SbTag>,
    ) -> Result<Option<usize>, ()> {
        #[cfg(feature = "expensive-debug-assertions")]
        self.verify_cache_consistency();

        let SbTagExtra::Concrete(tag) = tag else {
            // Handle the wildcard case.
            // Go search the stack for an exposed tag.
            if let Some(idx) =
                self.borrows
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
        let found = self.unknown_bottom.is_some_and(|&unknown_limit| {
            tag.0 < unknown_limit.0 // unknown_limit is an upper bound for what can be in the unknown bottom.
        });
        if found { Ok(None) } else { Err(()) }
    }

    fn find_granting_tagged(&mut self, access: AccessKind, tag: SbTag) -> Option<usize> {
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
    fn find_granting_cache(&mut self, access: AccessKind, tag: SbTag) -> Option<usize> {
        // This looks like a common-sense optimization; we're going to do a linear search of the
        // cache or the borrow stack to scan the shorter of the two. This optimization is miniscule
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
            // Make sure the possibly-unique range contains the new borrow
            self.unique_range.start = self.unique_range.start.min(new_idx);
            self.unique_range.end = self.unique_range.end.max(new_idx + 1);
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

        #[cfg(feature = "expensive-debug-assertions")]
        self.verify_cache_consistency();
    }

    /// Construct a new `Stack` using the passed `Item` as the base tag.
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

    #[allow(clippy::len_without_is_empty)] // Stacks are never empty
    pub fn len(&self) -> usize {
        self.borrows.len()
    }

    pub fn unknown_bottom(&self) -> Option<SbTag> {
        self.unknown_bottom
    }

    pub fn set_unknown_bottom(&mut self, tag: SbTag) {
        // We clear the borrow stack but the lookup cache doesn't support clearing per se. Instead,
        // there is a check explained in `find_granting_cache` which protects against accessing the
        // cache when it has been cleared and not yet refilled.
        self.borrows.clear();
        self.unknown_bottom = Some(tag);
    }

    /// Find all `Unique` elements in this borrow stack above `granting_idx`, pass a copy of them
    /// to the `visitor`, then set their `Permission` to `Disabled`.
    pub fn disable_uniques_starting_at<V: FnMut(Item) -> crate::InterpResult<'tcx>>(
        &mut self,
        disable_start: usize,
        mut visitor: V,
    ) -> crate::InterpResult<'tcx> {
        #[cfg(feature = "stack-cache")]
        let unique_range = self.unique_range.clone();
        #[cfg(not(feature = "stack-cache"))]
        let unique_range = 0..self.len();

        if disable_start <= unique_range.end {
            let lower = unique_range.start.max(disable_start);
            let upper = (unique_range.end + 1).min(self.borrows.len());
            for item in &mut self.borrows[lower..upper] {
                if item.perm() == Permission::Unique {
                    log::trace!("access: disabling item {:?}", item);
                    visitor(*item)?;
                    item.set_permission(Permission::Disabled);
                    // Also update all copies of this item in the cache.
                    for it in &mut self.cache.items {
                        if it.tag() == item.tag() {
                            it.set_permission(Permission::Disabled);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "stack-cache")]
        if disable_start < self.unique_range.start {
            // We disabled all Unique items
            self.unique_range.start = 0;
            self.unique_range.end = 0;
        } else {
            // Truncate the range to disable_start. This is + 2 because we are only removing
            // elements after disable_start, and this range does not include the end.
            self.unique_range.end = self.unique_range.end.min(disable_start + 1);
        }

        #[cfg(feature = "expensive-debug-assertions")]
        self.verify_cache_consistency();

        Ok(())
    }

    /// Produces an iterator which iterates over `range` in reverse, and when dropped removes that
    /// range of `Item`s from this `Stack`.
    pub fn pop_items_after<V: FnMut(Item) -> crate::InterpResult<'tcx>>(
        &mut self,
        start: usize,
        mut visitor: V,
    ) -> crate::InterpResult<'tcx> {
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
            // keep track of how many invalid elements there are and overwrite them with the base tag.
            // The base tag here serves as a harmless default value.
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

            if start < self.unique_range.start.saturating_sub(1) {
                // We removed all the Unique items
                self.unique_range = 0..0;
            } else {
                // Ensure the range doesn't extend past the new top of the stack
                self.unique_range.end = self.unique_range.end.min(start + 1);
            }
        } else {
            self.unique_range = 0..0;
        }

        #[cfg(feature = "expensive-debug-assertions")]
        self.verify_cache_consistency();
        Ok(())
    }
}
