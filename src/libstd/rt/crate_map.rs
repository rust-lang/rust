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
use iter::Iterator;
use option::{Some, None, Option};
use ptr::RawPtr;
use vec::ImmutableVector;
use rt::rtio::EventLoop;

// Need to tell the linker on OS X to not barf on undefined symbols
// and instead look them up at runtime, which we need to resolve
// the crate_map properly.
#[cfg(target_os = "macos")]
#[link_args = "-Wl,-U,__rust_crate_map_toplevel"]
extern {}

pub struct ModEntry<'a> {
    name: &'a str,
    log_level: *mut u32
}

pub struct CrateMap<'a> {
    version: i32,
    entries: &'a [ModEntry<'a>],
    children: &'a [&'a CrateMap<'a>],
    event_loop_factory: Option<fn() -> ~EventLoop>,
}

#[cfg(not(windows))]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    extern {
        #[crate_map]
        static CRATE_MAP: CrateMap<'static>;
    }

    let ptr: (*CrateMap) = &'static CRATE_MAP;
    if ptr.is_null() {
        return None;
    } else {
        return Some(&'static CRATE_MAP);
    }
}

#[cfg(windows)]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    use cast::transmute;
    use c_str::ToCStr;
    use unstable::dynamic_lib::dl;

    let sym = unsafe {
        let module = dl::open_internal();
        let rust_crate_map_toplevel = if cfg!(target_arch = "x86") {
            "__rust_crate_map_toplevel"
        } else {
            "_rust_crate_map_toplevel"
        };
        let sym = rust_crate_map_toplevel.with_c_str(|buf| {
            dl::symbol(module, buf)
        });
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
        _ => return 0
    }
}

fn do_iter_crate_map<'a>(
                     crate_map: &'a CrateMap<'a>,
                     f: |&ModEntry|,
                     visited: &mut HashSet<*CrateMap<'a>>) {
    if visited.insert(crate_map as *CrateMap) {
        match version(crate_map) {
            2 => {
                let (entries, children) = (crate_map.entries, crate_map.children);
                for entry in entries.iter() {
                    f(entry);
                }
                for child in children.iter() {
                    do_iter_crate_map(*child, |x| f(x), visited);
                }
            },
            _ => fail!("invalid crate map version")
        }
    }
}

/// Iterates recursively over `crate_map` and all child crate maps
pub fn iter_crate_map<'a>(crate_map: &'a CrateMap<'a>, f: |&ModEntry|) {
    // XXX: use random numbers as keys from the OS-level RNG when there is a nice
    //        way to do this
    let mut v: HashSet<*CrateMap<'a>> = HashSet::with_capacity_and_keys(0, 0, 32);
    do_iter_crate_map(crate_map, f, &mut v);
}

#[cfg(test)]
mod tests {
    use option::None;
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
            children: [],
            event_loop_factory: None,
        };

        let root_crate = CrateMap {
            version: 2,
            entries: [],
            children: [&child_crate, &child_crate],
            event_loop_factory: None,
        };

        let mut cnt = 0;
        unsafe {
            iter_crate_map(&root_crate, |entry| {
                assert!(*entry.log_level == 3);
                cnt += 1;
            });
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
            children: [],
            event_loop_factory: None,
        };

        let child_crate1 = CrateMap {
            version: 2,
            entries: [
                ModEntry { name: "t::f1", log_level: &mut 1},
            ],
            children: [&child_crate2],
            event_loop_factory: None,
        };

        let root_crate = CrateMap {
            version: 2,
            entries: [
                ModEntry { name: "t::f2", log_level: &mut 0},
            ],
            children: [&child_crate1],
            event_loop_factory: None,
        };

        let mut cnt = 0;
        unsafe {
            iter_crate_map(&root_crate, |entry| {
                assert!(*entry.log_level == cnt);
                cnt += 1;
            });
            assert!(cnt == 4);
        }
    }
}
