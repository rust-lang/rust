// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(stage0))] use cast::transmute;
use container::MutableSet;
use hashmap::HashSet;
use option::{Some, None};
use vec::ImmutableVector;

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

pub struct ModEntry<'self> {
    name: &'self str,
    log_level: *mut u32
}

pub struct CrateMapV0<'self> {
    entries: &'self [ModEntry<'self>],
    children: &'self [&'self CrateMap<'self>]
}

pub struct CrateMap<'self> {
    version: i32,
    entries: &'self [ModEntry<'self>],
    /// a dynamically sized struct, where all pointers to children are listed adjacent
    /// to the struct, terminated with NULL
    children: &'self [&'self CrateMap<'self>]
}

#[cfg(not(windows))]
pub fn get_crate_map() -> &'static CrateMap<'static> {
    &'static CRATE_MAP
}

#[cfg(windows)]
#[fixed_stack_segment]
#[inline(never)]
pub fn get_crate_map() -> &'static CrateMap<'static> {
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
    sym
}

fn version(crate_map: &CrateMap) -> i32 {
    match crate_map.version {
        1 => return 1,
        _ => return 0
    }
}

#[cfg(not(stage0))]
fn get_entries_and_children<'a>(crate_map: &'a CrateMap<'a>) ->
                    (&'a [ModEntry<'a>], &'a [&'a CrateMap<'a>]) {
    match version(crate_map) {
        0 => {
            unsafe {
                let v0: &'a CrateMapV0<'a> = transmute(crate_map);
                return (v0.entries, v0.children);
            }
        }
        1 => return (*crate_map).entries,
        _ => fail2!("Unknown crate map version!")
    }
}

#[cfg(not(stage0))]
fn iter_module_map(mod_entries: &[ModEntry], f: &fn(&ModEntry)) {
    for entry in mod_entries.iter() {
        f(entry);
    }
}

#[cfg(not(stage0))]
fn do_iter_crate_map<'a>(crate_map: &'a CrateMap<'a>, f: &fn(&ModEntry),
                            visited: &mut HashSet<*CrateMap<'a>>) {
    if visited.insert(crate_map as *CrateMap) {
        let (entries, children) = get_entries_and_children(crate_map);
        iter_module_map(entries, |x| f(x));
        for child in children.iter() {
            do_iter_crate_map(*child, |x| f(x), visited);
        }
    }
}

#[cfg(stage0)]
/// Iterates recursively over `crate_map` and all child crate maps
pub fn iter_crate_map<'a>(crate_map: &'a CrateMap<'a>, f: &fn(&ModEntry)) {
}

#[cfg(not(stage0))]
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
            version: 1,
            entries: entries,
            children: []
        };

        let root_crate = CrateMap {
            version: 1,
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
            version: 1,
            entries: [
                ModEntry { name: "c::m1", log_level: &mut level2},
                ModEntry { name: "c::m2", log_level: &mut level3},
            ],
            children: []
        };

        let child_crate1 = CrateMap {
            version: 1,
            entries: [
                ModEntry { name: "t::f1", log_level: &mut 1},
            ],
            children: [&child_crate2]
        };

        let root_crate = CrateMap {
            version: 1,
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
}
