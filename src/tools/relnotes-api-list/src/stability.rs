//! Walk the JSON to determine the stability of every item.
//!
//! There are two ways stability can be assigned to an item: either directly (with the `#[stable]`
//! or `#[unstable]` attributes), or inherited from its parent. The latter case is often used when a
//! `#![unstable]` attribute is applied to the whole module to mark all of its members as unstable.
//!
//! While determining directly assigned stability is trivial and can be done as part of the main
//! conversion visitor, determining inherited stability requires a separate visitor pass. Let's take
//! the following code as an example:
//!
//! ```rust,ignore
//! pub use foo::*;
//!
//! mod foo {
//!     #![unstable(..)]
//!
//!     pub fn bar() {}
//!     pub fn baz() {}
//! }
//! ```
//!
//! In this case, the conversion pass would visit the `pub use`, follow the import, and then visit
//! `bar()` and `baz()` (without first visiting `foo`). The `pub use` doesn't have its own stability
//! attribute, and even if it had one, it shouldn't be propagated to the items it imports. So, the
//! visitor would not find the stability for `bar()` and `baz()`, even though they are unstable.
//!
//! To solve the problem, we first do a visit of every item to assign its stability, and _then_ we
//! do the final conversion. The stability visit would do nothing when following the import (as
//! there would be no inherited stability), but it would then determine the correct stability by
//! also visiting the `foo` module.

use std::collections::HashMap;
use std::mem::replace;

use anyhow::{Error, bail};
use rustdoc_json_types::{Id, ItemEnum};

use crate::store::{Store, StoreCrateId, StoreItem};
use crate::visitor::{Visitor, walk_item};

#[derive(Debug)]
pub(crate) struct StabilityStore {
    result: HashMap<(StoreCrateId, Id), Stability>,
}

impl StabilityStore {
    pub(crate) fn new() -> Self {
        Self { result: HashMap::new() }
    }

    pub(crate) fn add(&mut self, store: &Store, item: &StoreItem<'_>) -> Result<(), Error> {
        StabilityVisitor { store, result: &mut self.result, parent: None }.visit_item(item)?;
        Ok(())
    }

    pub(crate) fn get(&self, krate: StoreCrateId, item: Id) -> Option<Stability> {
        self.result.get(&(krate, item)).copied()
    }
}

struct StabilityVisitor<'a, 'b> {
    store: &'a Store,
    result: &'b mut HashMap<(StoreCrateId, Id), Stability>,

    parent: Option<Stability>,
}

impl<'a> Visitor<'a> for StabilityVisitor<'a, '_> {
    type Result = ();

    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<(), Error> {
        let mut restore_stability = None;
        if let Some(stability) = parse_stability("Stability", &item.attrs)?.or(self.parent) {
            self.result.insert((item.crate_id, item.id), stability);

            restore_stability = Some(match &item.inner {
                // When we are traversing through an `use` we erase the current stability, as stability
                // is not inherited through `use`s.
                ItemEnum::Use(_) => replace(&mut self.parent, None),
                _ => replace(&mut self.parent, Some(stability)),
            });
        }

        walk_item(self, item)?;

        if let Some(restore) = restore_stability {
            self.parent = restore;
        }

        Ok(())
    }

    fn store(&self) -> &'a Store {
        self.store
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Stability {
    Stable,
    Unstable,
}

fn parse_stability(
    attribute_name: &str,
    attributes: &[String],
) -> Result<Option<Stability>, Error> {
    let attribute_prefix = format!("#[attr = {attribute_name}");
    for attribute in attributes {
        if !attribute.starts_with(&attribute_prefix) {
            continue;
        }

        if attribute.contains("level: Stable") || attribute.contains("level:\nStable") {
            return Ok(Some(Stability::Stable));
        } else if attribute.contains("level: Unstable") || attribute.contains("level:\nUnstable") {
            return Ok(Some(Stability::Unstable));
        } else {
            bail!("couldn't parse stability attribute: {attribute}");
        }
    }
    Ok(None)
}
