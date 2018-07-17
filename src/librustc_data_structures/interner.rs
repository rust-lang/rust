// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;
use std::hash::Hash;
use std::hash::Hasher;
use std::hash::BuildHasher;
use std::mem::{self, size_of};
use std::ptr::{Unique, NonNull};
use std::alloc::{Global, Alloc};
use std::collections::hash_map::RandomState;
use std::borrow::Borrow;
use std::fmt;

const ENTRIES_PER_GROUP: usize = 5;

#[repr(align(64), C)]
pub struct Group {
    hashes: [u32; ENTRIES_PER_GROUP],
    size: u32,
    values: [u64; ENTRIES_PER_GROUP],
}

impl Group {
    #[inline(always)]
    fn search_for_empty(&self) -> Option<usize> {
        if self.size != ENTRIES_PER_GROUP as u32 {
            Some(self.size as usize)
        } else {
            None
        }
    }

    #[inline(always)]
    fn search_with<K, F: FnMut(&K) -> bool>(&self, eq: &mut F, hash: u32) -> Option<(usize, bool)> {
        for i in 0..ENTRIES_PER_GROUP {
            let h = unsafe { *self.hashes.get_unchecked(i) };
            if h == hash && eq(unsafe { mem::transmute(self.values.get_unchecked(i)) }) {
                return Some((i, false))
            }
        }
        self.search_for_empty().map(|i| (i, true))
    }

    #[inline(always)]
    fn set(&mut self, pos: usize, hash: u32, value: u64) {
        unsafe {
            *self.hashes.get_unchecked_mut(pos) = hash;
            *self.values.get_unchecked_mut(pos) = value;
        }
    }

    #[inline(always)]
    fn iter<F: FnMut(u32, u64)>(&self, f: &mut F) {
        for i in 0..ENTRIES_PER_GROUP {
            unsafe {
                let h = *self.hashes.get_unchecked(i);
                if h != 0 {
                    f(h, *self.values.get_unchecked(i))
                }
            }
        }
    }
}

pub struct Table {
    group_mask: usize,
    size: usize,
    capacity: usize,
    groups: Unique<Group>,
}

pub struct RawEntry {
    group: *mut Group,
    pos: usize,
    empty: bool
}

impl Drop for Table {
    fn drop(&mut self) {
        if self.group_mask == 0 {
            return;
        }

        unsafe {
            Global.dealloc_array(
                NonNull::new_unchecked(self.groups.as_ptr()),
                self.group_mask + 1
            ).unwrap();
        }
    }
}

impl Table {
    fn new(group_count: usize) -> Table {
        assert!(size_of::<Group>() == 64);
        let groups: NonNull<Group> = Global.alloc_array(group_count).unwrap();
        let capacity2 = group_count * ENTRIES_PER_GROUP;
        let capacity1 = capacity2 - 1;
        //let capacity = (capacity1 * 10 + 10 - 1) / 11;
        let capacity = (capacity1 * 10 + 10 - 1) / 13;
        //println!("capacity1 {} capacity {}", capacity1, capacity);
        assert!(capacity < capacity2);

        for i in 0..group_count {
            let group = unsafe {
                &mut (*groups.as_ptr().offset(i as isize))
            };
            group.hashes = [0; ENTRIES_PER_GROUP];
            group.size = 0;
        }

        Table {
            group_mask: group_count.wrapping_sub(1),
            size: 0,
            capacity,
            groups: unsafe { Unique::new_unchecked(groups.as_ptr()) },
        }
    }

    fn search_for_empty(&self, hash: u64) -> RawEntry {
        let group_idx = hash as u32 as usize;
        let mask = self.group_mask;
        let mut group_idx = group_idx & mask;

        loop {
            let group_ptr = unsafe {
                self.groups.as_ptr().offset(group_idx as isize)
            };
            let group = unsafe {
                &(*group_ptr)
            };
            match group.search_for_empty() {
                Some(pos) => return RawEntry {
                    group: group_ptr,
                    pos,
                    empty: true,
                },
                None => (),
            }
            group_idx = (group_idx + 1) & mask;
        }
    }

    fn search_with<K, F: FnMut(&K) -> bool>(&self, mut eq: F, hash: u64) -> RawEntry {
        let group_idx = hash as u32 as usize;
        let mask = self.group_mask;
        let mut group_idx = group_idx & mask;

        loop {
            let group_ptr = unsafe {
                self.groups.as_ptr().offset(group_idx as isize)
            };
            let group = unsafe {
                &(*group_ptr)
            };
            let r = group.search_with(&mut eq, hash as u32);
            match r {
                Some((pos, empty)) => return RawEntry {
                    group: group_ptr,
                    pos,
                    empty,
                },
                None => (),
            }
            group_idx = (group_idx + 1) & mask;
        }
    }

    fn iter<F: FnMut(u32, u64)>(&self, mut f: F) {
        if self.group_mask == 0 {
            return;
        }
        for i in 0..(self.group_mask + 1) {
            let group = unsafe {
                &(*self.groups.as_ptr().offset(i as isize))
            };
            group.iter(&mut f);
        }
    }
}

pub struct Interner<K: Eq + Hash, S = RandomState> {
    hash_builder: S,
    table: Table,
    marker: PhantomData<K>,
}

impl<K: Eq + Hash, S> fmt::Debug for Interner<K, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        "Interner".fmt(f)
    }
}

impl<K: Eq + Hash, S: Default> Default for Interner<K, S> {
    fn default() -> Self {
        assert!(size_of::<K>() == 8);
        Interner {
            hash_builder: S::default(),
            table: Table {
                group_mask: 0,
                size: 0,
                capacity: 0,
                groups: unsafe { Unique::new_unchecked(NonNull::dangling().as_ptr()) },
            },
            marker: PhantomData,
        }
    }
}

pub fn make_hash<T: ?Sized, S>(hash_state: &S, t: &T) -> u64
    where T: Hash,
          S: BuildHasher
{
    let mut state = hash_state.build_hasher();
    t.hash(&mut state);
    state.finish() | (1 << 31)
}

impl<K: Eq + Hash, S: BuildHasher> Interner<K, S> {
    #[inline(never)]
    #[cold]
    fn expand(&mut self) {
        let mut new_table = Table::new((self.table.group_mask + 1) << 1);
        new_table.size = self.table.size;
        self.table.iter(|h, v| {
            let spot = new_table.search_for_empty(h as u64);
            unsafe {
                (*spot.group).size += 1;
                (*spot.group).set(spot.pos, h, v);
            }
        });
        self.table = new_table;
    }

    #[inline(always)]
    fn incr(&mut self) {
        if self.table.size + 1 > self.table.capacity {
            self.expand()
        }
    }

    pub fn len(&self) -> usize {
        self.table.size
    }

    #[inline]
    pub fn intern_ref<Q: ?Sized, F: FnOnce() -> K>(&mut self, value: &Q, make: F) -> &K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.incr();
        let hash = make_hash(&self.hash_builder, value);
        let spot = self.table.search_with::<K, _>(|k| value.eq(k.borrow()), hash);
        unsafe {
            if spot.empty {
                self.table.size += 1;
                (*spot.group).size += 1;
                let key = make();
                (*spot.group).set(spot.pos, hash as u32, *(&key as *const _ as *const u64));
            }
            &*((*spot.group).values.get_unchecked(spot.pos) as *const _ as *const K)
        }
    }

    #[inline]
    pub fn intern<Q, F: FnOnce(Q) -> K>(&mut self, value: Q, make: F) -> &K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.incr();
        let hash = make_hash(&self.hash_builder, &value);
        let spot = self.table.search_with::<K, _>(|k| value.eq(k.borrow()), hash);
        unsafe {
            if spot.empty {
                self.table.size += 1;
                (*spot.group).size += 1;
                let key = make(value);
                (*spot.group).set(spot.pos, hash as u32, *(&key as *const _ as *const u64));
            }
            &*((*spot.group).values.get_unchecked(spot.pos) as *const _ as *const K)
        }
    }
}
