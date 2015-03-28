// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module crawls a `clean::Crate` and produces a summarization of the
//! stability levels within the crate. The summary contains the module
//! hierarchy, with item counts for every stability level per module. A parent
//! module's count includes its children's.

use std::cmp::Ordering;
use std::ops::Add;

use syntax::attr::{Unstable, Stable};
use syntax::ast::Public;

use clean::{Crate, Item, ModuleItem, Module, EnumItem, Enum};
use clean::{ImplItem, Impl, Trait, TraitItem};
use clean::{ExternCrateItem, ImportItem, PrimitiveItem, Stability};

use html::render::cache;

#[derive(RustcEncodable, RustcDecodable, PartialEq, Eq)]
/// The counts for each stability level.
#[derive(Copy)]
pub struct Counts {
    pub deprecated: u64,
    pub unstable: u64,
    pub stable: u64,

    /// No stability level, inherited or otherwise.
    pub unmarked: u64,
}

impl Add for Counts {
    type Output = Counts;

    fn add(self, other: Counts) -> Counts {
        Counts {
            deprecated:   self.deprecated   + other.deprecated,
            unstable:     self.unstable     + other.unstable,
            stable:       self.stable       + other.stable,
            unmarked:     self.unmarked     + other.unmarked,
        }
    }
}

impl Counts {
    fn zero() -> Counts {
        Counts {
            deprecated:   0,
            unstable:     0,
            stable:       0,
            unmarked:     0,
        }
    }

    pub fn total(&self) -> u64 {
        self.deprecated + self.unstable + self.stable + self.unmarked
    }
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, Eq)]
/// A summarized module, which includes total counts and summarized children
/// modules.
pub struct ModuleSummary {
    pub name: String,
    pub counts: Counts,
    pub submodules: Vec<ModuleSummary>,
}

impl PartialOrd for ModuleSummary {
    fn partial_cmp(&self, other: &ModuleSummary) -> Option<Ordering> {
        self.name.partial_cmp(&other.name)
    }
}

impl Ord for ModuleSummary {
    fn cmp(&self, other: &ModuleSummary) -> Ordering {
        self.name.cmp(&other.name)
    }
}

// is the item considered publicly visible?
fn visible(item: &Item) -> bool {
    match item.inner {
        ImplItem(_) => true,
        _ => item.visibility == Some(Public)
    }
}

fn count_stability(stab: Option<&Stability>) -> Counts {
    match stab {
        None            => Counts { unmarked: 1,     .. Counts::zero() },
        Some(ref stab) => {
            if !stab.deprecated_since.is_empty() {
                return Counts { deprecated: 1, .. Counts::zero() };
            }
            match stab.level {
                Unstable    => Counts { unstable: 1,     .. Counts::zero() },
                Stable      => Counts { stable: 1,       .. Counts::zero() },
            }
        }
    }
}

fn summarize_methods(item: &Item) -> Counts {
    match cache().impls.get(&item.def_id) {
        Some(v) => {
            v.iter().map(|i| {
                let count = count_stability(i.stability.as_ref());
                if i.impl_.trait_.is_none() {
                    count + i.impl_.items.iter()
                        .map(|ti| summarize_item(ti).0)
                        .fold(Counts::zero(), |acc, c| acc + c)
                } else {
                    count
                }
            }).fold(Counts::zero(), |acc, c| acc + c)
        },
        None => {
            Counts::zero()
        },
    }
}


// Produce the summary for an arbitrary item. If the item is a module, include a
// module summary. The counts for items with nested items (e.g. modules, traits,
// impls) include all children counts.
fn summarize_item(item: &Item) -> (Counts, Option<ModuleSummary>) {
    let item_counts = count_stability(item.stability.as_ref()) + summarize_methods(item);

    // Count this item's children, if any. Note that a trait impl is
    // considered to have no children.
    match item.inner {
        // Require explicit `pub` to be visible
        ImplItem(Impl { ref items, trait_: None, .. }) => {
            let subcounts = items.iter().filter(|i| visible(*i))
                                        .map(summarize_item)
                                        .map(|s| s.0)
                                        .fold(Counts::zero(), |acc, x| acc + x);
            (subcounts, None)
        }
        // `pub` automatically
        EnumItem(Enum { variants: ref subitems, .. }) => {
            let subcounts = subitems.iter().map(summarize_item)
                                           .map(|s| s.0)
                                           .fold(Counts::zero(), |acc, x| acc + x);
            (item_counts + subcounts, None)
        }
        TraitItem(Trait { ref items, .. }) => {
            let subcounts = items.iter().map(summarize_item)
                                        .map(|s| s.0)
                                        .fold(Counts::zero(), |acc, x| acc + x);
            (item_counts + subcounts, None)
        }
        ModuleItem(Module { ref items, .. }) => {
            let mut counts = item_counts;
            let mut submodules = Vec::new();

            for (subcounts, submodule) in items.iter().filter(|i| visible(*i))
                                                      .map(summarize_item) {
                counts = counts + subcounts;
                submodule.map(|m| submodules.push(m));
            }
            submodules.sort();

            (counts, Some(ModuleSummary {
                name: item.name.as_ref().map_or("".to_string(), |n| n.clone()),
                counts: counts,
                submodules: submodules,
            }))
        }
        // no stability information for the following items:
        ExternCrateItem(..) | ImportItem(_) |
        PrimitiveItem(_) => (Counts::zero(), None),
        _ => (item_counts, None)
    }
}

/// Summarizes the stability levels in a crate.
pub fn build(krate: &Crate) -> ModuleSummary {
    match krate.module {
        None => ModuleSummary {
            name: krate.name.clone(),
            counts: Counts::zero(),
            submodules: Vec::new(),
        },
        Some(ref item) => ModuleSummary {
            name: krate.name.clone(), .. summarize_item(item).1.unwrap()
        }
    }
}
