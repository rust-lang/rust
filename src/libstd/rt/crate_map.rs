// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use libc::{c_void, c_char};
use ptr;
use ptr::RawPtr;
use vec;
use hashmap::HashSet;
use container::MutableSet;

pub struct ModEntry{
    name: *c_char,
    log_level: *mut u32
}
struct CrateMapV0 {
    entries: *ModEntry,
    children: [*CrateMap, ..1]
}

struct CrateMap {
    version: i32,
    annihilate_fn: *c_void,
    entries: *ModEntry,
    /// a dynamically sized struct, where all pointers to children are listed adjacent
    /// to the struct, terminated with NULL
    children: [*CrateMap, ..1]
}

unsafe fn version(crate_map: *CrateMap) -> i32 {
    match (*crate_map).version {
        1 => return 1,
        _ => return 0
    }
}

/// Returns a pointer to the annihilate function of the CrateMap
pub unsafe fn annihilate_fn(crate_map: *CrateMap) -> *c_void {
    match version(crate_map) {
        0 => return ptr::null(),
        1 => return (*crate_map).annihilate_fn,
        _ => fail!("Unknown crate map version!")
    }
}

unsafe fn entries(crate_map: *CrateMap) -> *ModEntry {
    match version(crate_map) {
        0 => {
            let v0 = crate_map as (*CrateMapV0);
            return (*v0).entries;
        }
        1 => return (*crate_map).entries,
        _ => fail!("Unknown crate map version!")
    }
}

unsafe fn iterator(crate_map: *CrateMap) -> **CrateMap {
    match version(crate_map) {
        0 => {
            let v0 = crate_map as (*CrateMapV0);
            return vec::raw::to_ptr((*v0).children);
        }
        1 => return vec::raw::to_ptr((*crate_map).children),
        _ => fail!("Unknown crate map version!")
    }
}

unsafe fn iter_module_map(mod_entries: *ModEntry, f: &fn(*mut ModEntry)) {
    let mut curr = mod_entries;

    while !(*curr).name.is_null() {
        f(curr as *mut ModEntry);
        curr = curr.offset(1);
    }
}

unsafe fn do_iter_crate_map(crate_map: *CrateMap, f: &fn(*mut ModEntry),
                            visited: &mut HashSet<*CrateMap>) {
    if visited.insert(crate_map) {
        iter_module_map(entries(crate_map), |x| f(x));
        let child_crates = iterator(crate_map);
        do ptr::array_each(child_crates) |child| {
            do_iter_crate_map(child, |x| f(x), visited);
        }
    }
}

/// Iterates recursively over `crate_map` and all child crate maps
pub unsafe fn iter_crate_map(crate_map: *CrateMap, f: &fn(*mut ModEntry)) {
    // XXX: use random numbers as keys from the OS-level RNG when there is a nice
    //        way to do this
    let mut v: HashSet<*CrateMap> = HashSet::with_capacity_and_keys(0, 0, 32);
    do_iter_crate_map(crate_map, f, &mut v);
}

#[test]
fn iter_crate_map_duplicates() {
    use c_str::ToCStr;
    use cast::transmute;

    struct CrateMapT3 {
        version: i32,
        annihilate_fn: *c_void,
        entries: *ModEntry,
        children: [*CrateMap, ..3]
    }

    unsafe {
        let mod_name1 = "c::m1".to_c_str();
        let mut level3: u32 = 3;

        let entries: ~[ModEntry] = ~[
            ModEntry { name: mod_name1.with_ref(|buf| buf), log_level: &mut level3},
            ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
        ];
        let child_crate = CrateMap {
            version: 1,
            annihilate_fn: ptr::null(),
            entries: vec::raw::to_ptr(entries),
            children: [ptr::null()]
        };

        let root_crate = CrateMapT3 {
            version: 1, annihilate_fn: ptr::null(),
            entries: vec::raw::to_ptr([ModEntry { name: ptr::null(), log_level: ptr::mut_null()}]),
            children: [&child_crate as *CrateMap, &child_crate as *CrateMap, ptr::null()]
        };

        let mut cnt = 0;
        do iter_crate_map(transmute(&root_crate)) |entry| {
            assert!(*(*entry).log_level == 3);
            cnt += 1;
        }
        assert!(cnt == 1);
    }
}

#[test]
fn iter_crate_map_follow_children() {
    use c_str::ToCStr;
    use cast::transmute;

    struct CrateMapT2 {
        version: i32,
        annihilate_fn: *c_void,
        entries: *ModEntry,
        children: [*CrateMap, ..2]
    }

    unsafe {
        let mod_name1 = "c::m1".to_c_str();
        let mod_name2 = "c::m2".to_c_str();
        let mut level2: u32 = 2;
        let mut level3: u32 = 3;
        let child_crate2 = CrateMap {
            version: 1,
            annihilate_fn: ptr::null(),
            entries: vec::raw::to_ptr([
                ModEntry { name: mod_name1.with_ref(|buf| buf), log_level: &mut level2},
                ModEntry { name: mod_name2.with_ref(|buf| buf), log_level: &mut level3},
                ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
            ]),
            children: [ptr::null()]
        };

        let child_crate1 = CrateMapT2 {
            version: 1,
            annihilate_fn: ptr::null(),
            entries: vec::raw::to_ptr([
                ModEntry { name: "t::f1".to_c_str().with_ref(|buf| buf), log_level: &mut 1},
                ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
            ]),
            children: [&child_crate2 as *CrateMap, ptr::null()]
        };

        let child_crate1_ptr: *CrateMap = transmute(&child_crate1);
        let root_crate = CrateMapT2 {
            version: 1, annihilate_fn: ptr::null(),
            entries: vec::raw::to_ptr([
                ModEntry { name: "t::f1".to_c_str().with_ref(|buf| buf), log_level: &mut 0},
                ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
            ]),
            children: [child_crate1_ptr, ptr::null()]
        };

        let mut cnt = 0;
        do iter_crate_map(transmute(&root_crate)) |entry| {
            assert!(*(*entry).log_level == cnt);
            cnt += 1;
        }
        assert!(cnt == 4);
    }
}
