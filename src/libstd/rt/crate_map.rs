// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use container::MutableSet;
use hashmap::HashSet;
use option::{Some, None, Option};
use vec::ImmutableVector;

/// Imports for old crate map versions
use cast::transmute;
use libc::c_char;
use ptr;
use str::raw::from_c_str;
use vec;

// Need to tell the linker on OS X to not barf on undefined symbols
// and instead look them up at runtime, which we need to resolve
// the crate_map properly.
#[cfg(target_os = "macos")]
#[link_args = "-undefined dynamic_lookup"]
extern {}

#[cfg(not(windows))]
extern {
    #[weak_linkage]
    #[link_name = "_rust_crate_map_toplevel"]
    static CRATE_MAP: CrateMap<'static>;
}

/// structs for old crate map versions
pub struct ModEntryV0 {
    name: *c_char,
    log_level: *mut u32
}
pub struct CrateMapV0 {
    entries: *ModEntryV0,
    children: [*CrateMapV0, ..1]
}

pub struct CrateMapV1 {
    version: i32,
    entries: *ModEntryV0,
    /// a dynamically sized struct, where all pointers to children are listed adjacent
    /// to the struct, terminated with NULL
    children: [*CrateMapV1, ..1]
}

pub struct ModEntry<'self> {
    name: &'self str,
    log_level: *mut u32
}

pub struct CrateMap<'self> {
    version: i32,
    entries: &'self [ModEntry<'self>],
    children: &'self [&'self CrateMap<'self>]
}

#[cfg(not(windows))]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    let ptr: (*CrateMap) = &'static CRATE_MAP;
    if ptr.is_null() {
        return None;
    } else {
        return Some(&'static CRATE_MAP);
    }
}

#[cfg(windows)]
#[fixed_stack_segment]
#[inline(never)]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    use c_str::ToCStr;
    use unstable::dynamic_lib::dl;

    let sym = unsafe {
        let module = dl::open_internal();
        let sym = do "__rust_crate_map_toplevel".with_c_str |buf| {
            dl::symbol(module, buf)
        };
        dl::close(module);
        sym
    };
    let ptr: (*CrateMap) = sym as *CrateMap;
    if ptr.is_null() {
        return None;
    } else {
        unsafe {
            return Some(transmute(sym));
        }
    }
}

fn version(crate_map: &CrateMap) -> i32 {
    match crate_map.version {
        2 => return 2,
        1 => return 1,
        _ => return 0
    }
}

fn iter_module_map(mod_entries: &[ModEntry], f: &fn(&ModEntry)) {
    for entry in mod_entries.iter() {
        f(entry);
    }
}

unsafe fn iter_module_map_v0(entries: *ModEntryV0, f: &fn(&ModEntry)) {
    let mut curr = entries;
    while !(*curr).name.is_null() {
        let mod_entry = ModEntry { name: from_c_str((*curr).name), log_level: (*curr).log_level };
        f(&mod_entry);
        curr = curr.offset(1);
    }
}

fn do_iter_crate_map<'a>(crate_map: &'a CrateMap<'a>, f: &fn(&ModEntry),
                            visited: &mut HashSet<*CrateMap<'a>>) {
    if visited.insert(crate_map as *CrateMap) {
        match version(crate_map) {
            2 => {
                let (entries, children) = (crate_map.entries, crate_map.children);
                iter_module_map(entries, |x| f(x));
                for child in children.iter() {
                    do_iter_crate_map(*child, |x| f(x), visited);
                }
            },
            // code for old crate map versions
            1 => unsafe {
                let v1: *CrateMapV1 = transmute(crate_map);
                iter_module_map_v0((*v1).entries, |x| f(x));
                let children = vec::raw::to_ptr((*v1).children);
                do ptr::array_each(children) |child| {
                    do_iter_crate_map(transmute(child), |x| f(x), visited);
                }
            },
            0 => unsafe {
                let v0: *CrateMapV0 = transmute(crate_map);
                iter_module_map_v0((*v0).entries, |x| f(x));
                let children = vec::raw::to_ptr((*v0).children);
                do ptr::array_each(children) |child| {
                    do_iter_crate_map(transmute(child), |x| f(x), visited);
                }
            },
            _ => fail2!("invalid crate map version")
        }
    }
}

/// Iterates recursively over `crate_map` and all child crate maps
pub fn iter_crate_map<'a>(crate_map: &'a CrateMap<'a>, f: &fn(&ModEntry)) {
    // XXX: use random numbers as keys from the OS-level RNG when there is a nice
    //        way to do this
    let mut v: HashSet<*CrateMap<'a>> = HashSet::with_capacity_and_keys(0, 0, 32);
    do_iter_crate_map(crate_map, f, &mut v);
}

#[cfg(test)]
mod tests {
    use rt::crate_map::{CrateMap, ModEntry, iter_crate_map};

    #[test]
    fn iter_crate_map_duplicates() {
        let mut level3: u32 = 3;

        let entries = [
            ModEntry { name: "c::m1", log_level: &mut level3},
        ];

        let child_crate = CrateMap {
            version: 2,
            entries: entries,
            children: []
        };

        let root_crate = CrateMap {
            version: 2,
            entries: [],
            children: [&child_crate, &child_crate]
        };

        let mut cnt = 0;
        unsafe {
            do iter_crate_map(&root_crate) |entry| {
                assert!(*entry.log_level == 3);
                cnt += 1;
            }
            assert!(cnt == 1);
        }
    }

    #[test]
    fn iter_crate_map_follow_children() {
        let mut level2: u32 = 2;
        let mut level3: u32 = 3;
        let child_crate2 = CrateMap {
            version: 2,
            entries: [
                ModEntry { name: "c::m1", log_level: &mut level2},
                ModEntry { name: "c::m2", log_level: &mut level3},
            ],
            children: []
        };

        let child_crate1 = CrateMap {
            version: 2,
            entries: [
                ModEntry { name: "t::f1", log_level: &mut 1},
            ],
            children: [&child_crate2]
        };

        let root_crate = CrateMap {
            version: 2,
            entries: [
                ModEntry { name: "t::f2", log_level: &mut 0},
            ],
            children: [&child_crate1]
        };

        let mut cnt = 0;
        unsafe {
            do iter_crate_map(&root_crate) |entry| {
                assert!(*entry.log_level == cnt);
                cnt += 1;
            }
            assert!(cnt == 4);
        }
    }


    /// Tests for old crate map versions
    #[test]
    fn iter_crate_map_duplicates_v1() {
        use c_str::ToCStr;
        use cast::transmute;
        use ptr;
        use rt::crate_map::{CrateMapV1, ModEntryV0, iter_crate_map};
        use vec;

        struct CrateMapT3 {
            version: i32,
            entries: *ModEntryV0,
            children: [*CrateMapV1, ..3]
        }

        unsafe {
            let mod_name1 = "c::m1".to_c_str();
            let mut level3: u32 = 3;

            let entries: ~[ModEntryV0] = ~[
                ModEntryV0 { name: mod_name1.with_ref(|buf| buf), log_level: &mut level3},
                ModEntryV0 { name: ptr::null(), log_level: ptr::mut_null()}
            ];
            let child_crate = CrateMapV1 {
                version: 1,
                entries: vec::raw::to_ptr(entries),
                children: [ptr::null()]
            };

            let root_crate = CrateMapT3 {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntryV0 { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: [&child_crate as *CrateMapV1, &child_crate as *CrateMapV1, ptr::null()]
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
    fn iter_crate_map_follow_children_v1() {
        use c_str::ToCStr;
        use cast::transmute;
        use ptr;
        use rt::crate_map::{CrateMapV1, ModEntryV0, iter_crate_map};
        use vec;

        struct CrateMapT2 {
            version: i32,
            entries: *ModEntryV0,
            children: [*CrateMapV1, ..2]
        }

        unsafe {
            let mod_name1 = "c::m1".to_c_str();
            let mod_name2 = "c::m2".to_c_str();
            let mut level2: u32 = 2;
            let mut level3: u32 = 3;
            let child_crate2 = CrateMapV1 {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntryV0 { name: mod_name1.with_ref(|buf| buf), log_level: &mut level2},
                    ModEntryV0 { name: mod_name2.with_ref(|buf| buf), log_level: &mut level3},
                    ModEntryV0 { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: [ptr::null()]
            };

            let child_crate1 = CrateMapT2 {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntryV0 { name: "t::f1".with_c_str(|buf| buf), log_level: &mut 1},
                    ModEntryV0 { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: [&child_crate2 as *CrateMapV1, ptr::null()]
            };

            let child_crate1_ptr: *CrateMapV1 = transmute(&child_crate1);
            let root_crate = CrateMapT2 {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntryV0 { name: "t::f1".with_c_str(|buf| buf), log_level: &mut 0},
                    ModEntryV0 { name: ptr::null(), log_level: ptr::mut_null()}
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
}
