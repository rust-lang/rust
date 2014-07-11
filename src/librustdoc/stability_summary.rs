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
//! module's count includes its childrens's.

use std::ops::Add;
use std::num::Zero;
use std::iter::AdditiveIterator;

use syntax::attr::{Deprecated, Experimental, Unstable, Stable, Frozen, Locked};
use syntax::ast::Public;

use clean::{Crate, Item, ModuleItem, Module, StructItem, Struct, EnumItem, Enum};
use clean::{ImplItem, Impl, TraitItem, Trait, TraitMethod, Provided, Required};
use clean::{ViewItemItem, PrimitiveItem};

#[deriving(Zero, Encodable, Decodable, PartialEq, Eq)]
/// The counts for each stability level.
pub struct Counts {
    pub deprecated: uint,
    pub experimental: uint,
    pub unstable: uint,
    pub stable: uint,
    pub frozen: uint,
    pub locked: uint,

    /// No stability level, inherited or otherwise.
    pub unmarked: uint,
}

impl Add<Counts, Counts> for Counts {
    fn add(&self, other: &Counts) -> Counts {
        Counts {
            deprecated:   self.deprecated   + other.deprecated,
            experimental: self.experimental + other.experimental,
            unstable:     self.unstable     + other.unstable,
            stable:       self.stable       + other.stable,
            frozen:       self.frozen       + other.frozen,
            locked:       self.locked       + other.locked,
            unmarked:     self.unmarked     + other.unmarked,
        }
    }
}

impl Counts {
    pub fn total(&self) -> uint {
        self.deprecated + self.experimental + self.unstable + self.stable +
            self.frozen + self.locked + self.unmarked
    }
}

#[deriving(Encodable, Decodable, PartialEq, Eq)]
/// A summarized module, which includes total counts and summarized chilcren
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

// is the item considered publically visible?
fn visible(item: &Item) -> bool {
    match item.inner {
        ImplItem(_) => true,
        _ => item.visibility == Some(Public)
    }
}

// Produce the summary for an arbitrary item. If the item is a module, include a
// module summary. The counts for items with nested items (e.g. modules, traits,
// impls) include all children counts.
fn summarize_item(item: &Item) -> (Counts, Option<ModuleSummary>) {
    // count this item
    let item_counts = match item.stability {
        None             => Counts { unmarked: 1,     .. Zero::zero() },
        Some(ref stab) => match stab.level {
            Deprecated   => Counts { deprecated: 1,   .. Zero::zero() },
            Experimental => Counts { experimental: 1, .. Zero::zero() },
            Unstable     => Counts { unstable: 1,     .. Zero::zero() },
            Stable       => Counts { stable: 1,       .. Zero::zero() },
            Frozen       => Counts { frozen: 1,       .. Zero::zero() },
            Locked       => Counts { locked: 1,       .. Zero::zero() },
        }
    };

    // Count this item's children, if any. Note that a trait impl is
    // considered to have no children.
    match item.inner {
        // Require explicit `pub` to be visible
        StructItem(Struct { fields: ref subitems, .. }) |
        ImplItem(Impl { methods: ref subitems, trait_: None, .. }) => {
            let subcounts = subitems.iter().filter(|i| visible(*i))
                                           .map(summarize_item)
                                           .map(|s| s.val0())
                                           .sum();
            (item_counts + subcounts, None)
        }
        // `pub` automatically
        EnumItem(Enum { variants: ref subitems, .. }) => {
            let subcounts = subitems.iter().map(summarize_item)
                                           .map(|s| s.val0())
                                           .sum();
            (item_counts + subcounts, None)
        }
        TraitItem(Trait { methods: ref methods, .. }) => {
            fn extract_item<'a>(meth: &'a TraitMethod) -> &'a Item {
                match *meth {
                    Provided(ref item) | Required(ref item) => item
                }
            }
            let subcounts = methods.iter().map(extract_item)
                                          .map(summarize_item)
                                          .map(|s| s.val0())
                                          .sum();
            (item_counts + subcounts, None)
        }
        ModuleItem(Module { items: ref items, .. }) => {
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
        ViewItemItem(_) | PrimitiveItem(_) => (Zero::zero(), None),
        _ => (item_counts, None)
    }
}

/// Summarizes the stability levels in a crate.
pub fn build(krate: &Crate) -> ModuleSummary {
    match krate.module {
        None => ModuleSummary {
            name: krate.name.clone(),
            counts: Zero::zero(),
            submodules: Vec::new(),
        },
        Some(ref item) => ModuleSummary {
            name: krate.name.clone(), .. summarize_item(item).val1().unwrap()
        }
    }
}
