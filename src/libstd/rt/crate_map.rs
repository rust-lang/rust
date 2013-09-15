// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast::transmute;
use container::MutableSet;
use hashmap::HashSet;
use libc::c_char;

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
    static CRATE_MAP: CrateMap;
}

pub struct ModEntry {
    name: *c_char,
    log_level: *mut u32
}

struct CrateMapV0 {
    entries: &static [ModEntry],
    children: &'static [&'static CrateMap]
}

struct CrateMap {
    version: i32,
    entries: &static [ModEntry],
    /// a dynamically sized struct, where all pointers to children are listed adjacent
    /// to the struct, terminated with NULL
    children: [*CrateMap, ..1]
}

#[cfg(not(windows))]
pub fn get_crate_map() -> *CrateMap {
    &'static CRATE_MAP as *CrateMap
}

#[cfg(windows)]
#[fixed_stack_segment]
#[inline(never)]
pub fn get_crate_map() -> *CrateMap {
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

    sym as *CrateMap
}

fn version(crate_map: &'static CrateMap) -> i32 {
    match crate_map.version {
        1 => return 1,
        _ => return 0
    }
}

#[cfg(not(stage0))]
fn entries(crate_map: &'static CrateMap) -> *ModEntry {
    match version(crate_map) {
        0 => {
            unsafe {
                let v0: &'static CrateMapV0 = transmute(crate_map);
                return v0.entries;
            }
        }
        1 => return (*crate_map).entries,
        _ => fail2!("Unknown crate map version!")
    }
}

#[cfg(not(stage0))]
fn iterator(crate_map: &'static CrateMap) -> &'static [&'static CrateMap] {
    match version(crate_map) {
        0 => {
            unsafe {
                let v0: &'static CrateMapV0 = transmute(crate_map);
                return v0.children;
            }
        }
        1 => return vec::raw::to_ptr((*crate_map).children),
        _ => fail2!("Unknown crate map version!")
    }
}

fn iter_module_map(mod_entries: *ModEntry, f: &fn(&mut ModEntry)) {
    let mut curr = mod_entries;

    unsafe {
        while !(*curr).name.is_null() {
            f(transmute(curr));
            curr = curr.offset(1);
        }
    }
}



#[cfg(not(stage0))]
fn do_iter_crate_map(crate_map: &'static CrateMap, f: &fn(&mut ModEntry),
                            visited: &mut HashSet<*CrateMap>) {
    if visited.insert(crate_map as *CrateMap) {
        iter_module_map(crate_map.entries, |x| f(x));
        let child_crates = iterator(crate_map);
        
        let mut i = 0;
        while i < child_crates.len() {
            do_iter_crate_map(child_crates[i], |x| f(x), visited);
            i = i + 1;
        }
    }
}

#[cfg(stage0)]
/// Iterates recursively over `crate_map` and all child crate maps
pub fn iter_crate_map(crate_map: *u8, f: &fn(&mut ModEntry)) {
}

#[cfg(not(stage0))]
/// Iterates recursively over `crate_map` and all child crate maps
pub fn iter_crate_map(crate_map: &'static CrateMap, f: &fn(&mut ModEntry)) {
    // XXX: use random numbers as keys from the OS-level RNG when there is a nice
    //        way to do this
    let mut v: HashSet<*CrateMap> = HashSet::with_capacity_and_keys(0, 0, 32);
    unsafe {
        do_iter_crate_map(transmute(crate_map), f, &mut v);
    }
}

#[cfg(test)]
mod tests {
    use c_str::ToCStr;
    use cast::transmute;
    use ptr;
    use vec;

    use rt::crate_map::{ModEntry, iter_crate_map};

    struct CrateMap<'self> { 
        version: i32,
        entries: *ModEntry,
        /// a dynamically sized struct, where all pointers to children are listed adjacent
        /// to the struct, terminated with NULL
        children: &'self [&'self CrateMap<'self>] 
    }

    #[test]
    fn iter_crate_map_duplicates() {
        unsafe {
            let mod_name1 = "c::m1".to_c_str();
            let mut level3: u32 = 3;

            let entries: ~[ModEntry] = ~[
                ModEntry { name: mod_name1.with_ref(|buf| buf), log_level: &mut level3},
                ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
            ];

            let child_crate = CrateMap {
                version: 1,
                entries: vec::raw::to_ptr(entries),
                children: []
            };

            let root_crate = CrateMap {
                version: 1,
                entries: vec::raw::to_ptr([ModEntry { name: ptr::null(), log_level: ptr::mut_null()}]),
                children: [&child_crate, &child_crate]
            };

            let mut cnt = 0;
            do iter_crate_map(transmute(&root_crate)) |entry| {
                assert!(*entry.log_level == 3);
                cnt += 1;
            }
            assert!(cnt == 1);
        }
    }

    #[test]
    fn iter_crate_map_follow_children() {
        unsafe {
            let mod_name1 = "c::m1".to_c_str();
            let mod_name2 = "c::m2".to_c_str();
            let mut level2: u32 = 2;
            let mut level3: u32 = 3;
            let child_crate2 = CrateMap {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntry { name: mod_name1.with_ref(|buf| buf), log_level: &mut level2},
                    ModEntry { name: mod_name2.with_ref(|buf| buf), log_level: &mut level3},
                    ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: []
            };

            let child_crate1 = CrateMap {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntry { name: "t::f1".to_c_str().with_ref(|buf| buf), log_level: &mut 1},
                    ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: [&child_crate2]
            };

            let root_crate = CrateMap {
                version: 1,
                entries: vec::raw::to_ptr([
                    ModEntry { name: "t::f1".to_c_str().with_ref(|buf| buf), log_level: &mut 0},
                    ModEntry { name: ptr::null(), log_level: ptr::mut_null()}
                ]),
                children: [&child_crate1]
            };

            let mut cnt = 0;
            do iter_crate_map(transmute(&root_crate)) |entry| {
                assert!(*entry.log_level == cnt);
                cnt += 1;
            }
            assert!(cnt == 4);
        }
    }
}
