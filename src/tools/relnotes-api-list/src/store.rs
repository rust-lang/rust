use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;

use anyhow::{Error, anyhow, bail};
use rustdoc_json_types::{Crate, Id, Item, ItemEnum, ItemKind};

use crate::visitor::is_broken_use;

pub(crate) struct Store {
    crates: Vec<StoreCrate>,
    names: HashMap<String, StoreCrateId>,
}

impl Store {
    pub(crate) fn new() -> Self {
        Store { crates: Vec::new(), names: HashMap::new() }
    }

    pub(crate) fn load(&mut self, path: &Path) -> Result<StoreCrateId, Error> {
        let id = StoreCrateId(self.crates.len());
        let json: Crate = serde_json::from_slice(&std::fs::read(path)?)?;

        let name = json
            .index
            .get(&json.root)
            .ok_or_else(|| anyhow!("malformed json: root id is not in the index"))?
            .name
            .as_deref()
            .ok_or_else(|| anyhow!("malformed json: root node doesn't have a name"))?
            .to_string();

        self.names.insert(name.clone(), id);
        self.crates.push(StoreCrate {
            ids: json
                .paths
                .iter()
                .map(|(id, path)| ((path.kind, path.path.clone()), *id))
                .collect(),
            json,
            name,
        });

        Ok(id)
    }

    pub(crate) fn item(&self, crate_id: StoreCrateId, item: Id) -> Result<StoreItem<'_>, Error> {
        let krate =
            self.crates.get(crate_id.0).ok_or_else(|| anyhow!("missing crate {crate_id:?}"))?;

        Ok(StoreItem {
            item: krate
                .json
                .index
                .get(&item)
                .ok_or_else(|| anyhow!("missing ID {item:?} in crate {}", krate.name))?,
            crate_id,
        })
    }

    pub(crate) fn crate_root(&self, id: StoreCrateId) -> Result<StoreItem<'_>, Error> {
        self.item(
            id,
            self.crates.get(id.0).ok_or_else(|| anyhow!("missing crate {id:?}"))?.json.root,
        )
    }

    pub(crate) fn crate_name(&self, id: StoreCrateId) -> &str {
        &self.crates[id.0].name
    }

    pub(crate) fn crate_ids(&self) -> impl Iterator<Item = StoreCrateId> {
        (0..self.crates.len()).map(|idx| StoreCrateId(idx))
    }

    pub(crate) fn resolve_cross_crate(
        &self,
        krate_id: StoreCrateId,
        item: Id,
    ) -> Result<UseDefinition, Error> {
        let krate =
            self.crates.get(krate_id.0).ok_or_else(|| anyhow!("missing crate {krate_id:?}"))?;

        // External IDs are defined in the `paths` map. Note that crate_id 0 is the current crate.
        // If this is a local item just return the same crate and item IDs.
        let Some(path) = krate.json.paths.get(&item).filter(|p| p.crate_id != 0) else {
            return Ok(UseDefinition { krate: krate_id, item });
        };

        let extern_summary = krate.json.external_crates.get(&path.crate_id).ok_or_else(|| {
            anyhow!("external crate ID {} not present in external_crates", path.crate_id)
        })?;
        let extern_store_id = self
            .names
            .get(&extern_summary.name)
            .ok_or_else(|| anyhow!("crate {} is not loaded by the too", extern_summary.name))?;
        let extern_crate = &self.crates[extern_store_id.0];

        if let Some(item) = extern_crate.ids.get(&(path.kind, path.path.clone())) {
            Ok(UseDefinition { krate: *extern_store_id, item: *item })
        } else {
            bail!("could not find item {path:?}");
        }
    }

    pub(crate) fn resolve_use_recursive(
        &self,
        krate_id: StoreCrateId,
        item: Id,
    ) -> Result<UseDefinition, Error> {
        let mut definition = UseDefinition { krate: krate_id, item };
        loop {
            if let ItemEnum::Use(use_) = &self.item(definition.krate, definition.item)?.inner {
                if is_broken_use(use_) {
                    break;
                }
                if let Some(use_id) = use_.id {
                    definition = self.resolve_cross_crate(definition.krate, use_id)?;
                    continue;
                }
            }
            break;
        }
        Ok(definition)
    }
}

struct StoreCrate {
    json: Crate,
    ids: HashMap<(ItemKind, Vec<String>), Id>,
    name: String,
}

#[derive(Debug)]
pub(crate) struct StoreItem<'a> {
    item: &'a Item,
    pub(crate) crate_id: StoreCrateId,
}

impl<'a> StoreItem<'a> {
    pub(crate) fn require_name(&self) -> Result<&'a str, Error> {
        self.item.name.as_deref().ok_or_else(|| anyhow!("no name for item {:?}", self.item.id))
    }
}

impl Deref for StoreItem<'_> {
    type Target = Item;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StoreCrateId(usize);

#[derive(Debug)]
pub(crate) struct UseDefinition {
    pub(crate) krate: StoreCrateId,
    pub(crate) item: Id,
}
