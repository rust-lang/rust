//! Walk the JSON to calculate the URL of each item.
//!
//! Calculating the URL for each item is harder than it might seem at first glance. Intuitively, you
//! would just walk the three, add segments to the URL as you visit, and then for each item finalize
//! the item from the segments you added.
//!
//! The naive approach works fine in most cases, and it could be implemented in `ConvertToSchema`
//! without having to create separate visitors. Unfortunately, it breaks down with `use`s.
//!
//! When an item is publicly re-exported with `pub use`, rustdoc can do one of two things:
//!
//! 1. If the item is also reachable somewhere else, rustdoc will just emit a `pub use` line in the
//!    "Re-exports" section of the module, linking to the source location of the item. As an
//!    example, the URL of `core::prelude::v1::Iterator` is `core/iter/trait.Iterator.html`.
//!
//! 2. If the item is not reachable anywhere else, rustdoc effectively inlines the item at the place
//!    imported, and the URL will point to the re-export. As an example, `MaybeUninit` is defined in
//!    the (private) `core::mem::maybe_uninit::MaybeUninit`, but it's re-exported in the (public)
//!    `core::mem::MaybeUninit`. Its URL is thus `core/mem/union.MaybeUninit.html`.
//!
//! To properly handle this case, when calculating the URLs we first need to know the set of all
//! items reachable without following `use`s. Then we can do another visiting pass to calculate all
//! URLs, using the reachability information to distinguish between cases 1 and 2.

use crate::store::{Store, StoreCrateId, StoreItem};
use crate::visitor::{Visitor, walk_item};
use anyhow::{Context, Error, anyhow, bail};
use rustdoc_json_types::{Id, ItemEnum, MacroKind};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct UrlStore {
    urls: HashMap<(StoreCrateId, Id), String>,
}

impl UrlStore {
    pub(crate) fn new(store: &Store, crates: &[StoreCrateId]) -> Result<Self, Error> {
        // First calculate whether items are reachable without following `use`s.
        let mut reachable = ReachableWithoutUse { store, reachable: HashSet::new() };
        for krate in crates {
            reachable.visit_item(&store.crate_root(*krate)?).with_context(|| {
                format!(
                    "failed to gather the items reachable without `use` in {}",
                    store.name_of_crate(*krate)
                )
            })?;
        }

        // ...and then use that information to assign the correct URLs.
        let mut assigner = AssignUrls {
            store,
            reachable_without_use: reachable.reachable,
            urls: HashMap::new(),
            stack: UrlStack::new(),
            within_use: false,
            within_impl: false,
        };
        for krate in crates {
            assigner.visit_item(&store.crate_root(*krate)?).with_context(|| {
                format!("failed to determine URLs in {}", store.name_of_crate(*krate))
            })?;
        }

        Ok(Self { urls: assigner.urls })
    }

    pub(crate) fn for_item(&self, item: &StoreItem<'_>) -> Result<&str, Error> {
        self.urls
            .get(&(item.crate_id, item.id))
            .ok_or_else(|| anyhow!("could not find URL for item {:?}", item.id))
            .map(|s| s.as_str())
    }
}

struct AssignUrls<'a> {
    store: &'a Store,
    reachable_without_use: HashSet<(StoreCrateId, Id)>,
    urls: HashMap<(StoreCrateId, Id), String>,

    stack: UrlStack,
    within_use: bool,
    within_impl: bool,
}

impl<'a> Visitor<'a> for AssignUrls<'a> {
    type Result = ();

    fn store(&self) -> &'a Store {
        self.store
    }

    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<(), Error> {
        let mut restore_within_use = None;
        let mut restore_within_impl = None;
        let mut should_pop = false;

        let mut push = |item: &StoreItem<'_>, component: UrlComponent| -> Result<(), Error> {
            self.stack.push(component);

            let id = (item.crate_id, item.id);
            if !self.within_use || !self.reachable_without_use.contains(&id) {
                self.urls.insert(id, self.stack.url()?);
            }
            Ok(())
        };
        let mut push_item = |kind: &str| -> Result<(), Error> {
            push(item, UrlComponent::Item(format!("{kind}.{}", item.require_name()?)))?;
            should_pop = true;
            Ok(())
        };

        match &item.inner {
            ItemEnum::AssocConst { .. } => push_item("associatedconstant")?,
            ItemEnum::AssocType { .. } => push_item("associatedtype")?,
            ItemEnum::Constant { .. } => push_item("constant")?,
            ItemEnum::Enum(_) => push_item("enum")?,
            ItemEnum::Macro(_) => push_item("macro")?,
            ItemEnum::Primitive(_) => push_item("primitive")?,
            ItemEnum::Static(_) => push_item("static")?,
            ItemEnum::Struct(_) => push_item("struct")?,
            ItemEnum::StructField(_) => push_item("structfield")?,
            ItemEnum::Trait(_) => push_item("trait")?,
            ItemEnum::TypeAlias(_) => push_item("type")?,
            ItemEnum::Union(_) => push_item("union")?,
            ItemEnum::Variant(_) => push_item("variant")?,

            ItemEnum::ProcMacro(proc_macro) => match proc_macro.kind {
                MacroKind::Bang => push_item("macro")?,
                MacroKind::Attr => push_item("attr")?,
                MacroKind::Derive => push_item("derive")?,
            },

            ItemEnum::Function(_) => {
                if self.within_impl {
                    push_item("method")?
                } else {
                    push_item("function")?
                }
            }

            ItemEnum::Module(_) => {
                push(item, UrlComponent::Dir(item.require_name()?.into()))?;
                should_pop = true;
            }

            ItemEnum::Use(_) => {
                restore_within_use = Some(self.within_use);
                self.within_use = true;
            }

            ItemEnum::Impl(_) => {
                // TODO: should we add an URL to impls?
                restore_within_impl = Some(self.within_impl);
                self.within_impl = true;
            }

            // Don't generate URLs for the following types:
            ItemEnum::ExternCrate { .. } => {}
            ItemEnum::TraitAlias(_) => {}
            ItemEnum::ExternType => {}
        }

        walk_item(self, item)?;

        if let Some(restore) = restore_within_use {
            self.within_use = restore;
        }
        if let Some(restore) = restore_within_impl {
            self.within_impl = restore;
        }
        if should_pop {
            self.stack.pop();
        }
        Ok(())
    }
}

struct ReachableWithoutUse<'a> {
    store: &'a Store,
    reachable: HashSet<(StoreCrateId, Id)>,
}

impl<'a> Visitor<'a> for ReachableWithoutUse<'a> {
    type Result = ();

    fn store(&self) -> &'a Store {
        self.store
    }

    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<(), Error> {
        if let ItemEnum::Use(_) = &item.inner {
            return Ok(());
        }

        self.reachable.insert((item.crate_id, item.id));
        walk_item(self, item)
    }
}

pub(crate) struct UrlStack {
    components: Vec<UrlComponent>,
}

impl UrlStack {
    pub(crate) fn new() -> Self {
        Self { components: Vec::new() }
    }

    pub(crate) fn push(&mut self, component: UrlComponent) {
        self.components.push(component);
    }

    pub(crate) fn pop(&mut self) {
        self.components.pop();
    }

    pub(crate) fn url(&self) -> Result<String, Error> {
        let mut state = UrlState::Dir;
        let mut url = String::new();
        for component in &self.components {
            match (state, component) {
                (UrlState::Dir, UrlComponent::Dir(dir)) => {
                    url.push_str(dir);
                    url.push('/');
                }
                (UrlState::Dir, UrlComponent::Item(item)) => {
                    url.push_str(item);
                    url.push_str(".html");
                    state = UrlState::Hash;
                }
                (UrlState::Hash, UrlComponent::Item(item)) => {
                    url.push('#');
                    url.push_str(item);
                    state = UrlState::End;
                }
                (_, UrlComponent::Dir(_)) => {
                    bail!("tried adding a dir after an item (url stack: {:?})", self.components)
                }
                (UrlState::End, UrlComponent::Item(_)) => {
                    bail!("adding after the end of the url (url stack: {:?})", self.components)
                }
            }
        }
        if let UrlState::Dir = state {
            url.push_str("index.html");
        }
        Ok(url)
    }
}

#[derive(Clone, Copy)]
enum UrlState {
    Dir,
    Hash,
    End,
}

#[derive(Debug)]
pub(crate) enum UrlComponent {
    Dir(String),
    Item(String),
}
