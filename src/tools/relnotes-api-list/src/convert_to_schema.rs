use crate::pretty_print::pretty_impl;
use crate::schema::SchemaItem;
use crate::stability::{Stability, StabilityStore};
use crate::store::{Store, StoreItem};
use crate::url::UrlStore;
use crate::visitor::{Visitor, walk_item};
use anyhow::{Error, bail};
use rustdoc_json_types::{ItemEnum, Visibility};

pub(crate) struct ConvertToSchema<'a> {
    store: &'a Store,
    stability: &'a StabilityStore,
    urls: &'a UrlStore,

    name_stack: Vec<String>,
}

impl<'a> Visitor<'a> for ConvertToSchema<'a> {
    type Result = Vec<SchemaItem>;

    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<Vec<SchemaItem>, Error> {
        if item.visibility != expected_visibility(item) {
            return Ok(Vec::new());
        }

        match (self.stability.get(item.crate_id, item.id), &item.inner) {
            (Some(Stability::Stable), _) => {}
            (Some(Stability::Unstable), _) => return Ok(Vec::new()),

            // `impl` blocks are not required to have stability attributes, so we should not error
            // out if they are missing them.
            (None, ItemEnum::Impl(_)) => {}

            (None, _) => bail!(
                "stability attribute is missing for {} {:?}",
                self.name_stack.join("::"),
                item.id
            ),
        }

        let mut pop_name = false;
        if let Some(name) = &item.name {
            self.name_stack.push(name.clone());
            pop_name = true;
        }

        let result = match &item.inner {
            ItemEnum::AssocConst { .. }
            | ItemEnum::AssocType { .. }
            | ItemEnum::Constant { .. }
            | ItemEnum::Enum(_)
            | ItemEnum::Macro(_)
            | ItemEnum::Module(_)
            | ItemEnum::Static(_)
            | ItemEnum::Struct(_)
            | ItemEnum::StructField(_)
            | ItemEnum::Trait(_)
            | ItemEnum::TypeAlias(_)
            | ItemEnum::Union(_)
            | ItemEnum::Variant(_)
            | ItemEnum::ProcMacro(_)
            | ItemEnum::Function(_) => self.include(item),

            ItemEnum::Primitive(p) => {
                // We don't want primitives to have the `std::` prefix.
                let old_stack = std::mem::replace(&mut self.name_stack, vec![p.name.clone()]);
                let result = self.include(item);
                self.name_stack = old_stack;
                result
            }

            ItemEnum::Use(_) => walk_item(self, item),

            ItemEnum::Impl(impl_) => {
                if impl_.trait_.is_some() {
                    Ok(vec![SchemaItem {
                        name: pretty_impl(impl_),
                        deprecated: item.deprecation.is_some(),
                        url: None, // TODO: is there an URL we can put here?

                        // We are intentionally not walking inside impls of traits: we don't want
                        // all types in the standard library to show up in the changelog if a new
                        // item is added in a trait.
                        children: Vec::new(),
                    }])
                } else {
                    walk_item(self, item)
                }
            }

            ItemEnum::TraitAlias(_) | ItemEnum::ExternType | ItemEnum::ExternCrate { .. } => {
                Ok(Vec::new())
            }
        };

        if pop_name {
            self.name_stack.pop();
        }
        result
    }

    fn store(&self) -> &'a Store {
        self.store
    }
}

impl<'a> ConvertToSchema<'a> {
    pub(crate) fn new(store: &'a Store, stability: &'a StabilityStore, urls: &'a UrlStore) -> Self {
        Self { store, stability, urls, name_stack: Vec::new() }
    }

    fn include(&mut self, item: &StoreItem<'a>) -> Result<Vec<SchemaItem>, Error> {
        let item = SchemaItem {
            name: self.name_stack.join("::"),
            url: Some(self.urls.for_item(item)?.into()),
            deprecated: item.deprecation.is_some(),
            children: walk_item(self, item)?,
        };
        Ok(vec![item])
    }
}

/// Some items don't have a visibility associated to them, and are instead public by default. This
/// function determines what visibility a public item must have.
fn expected_visibility(item: &StoreItem<'_>) -> Visibility {
    match &item.inner {
        ItemEnum::AssocType { .. }
        | ItemEnum::AssocConst { .. }
        | ItemEnum::Variant(_)
        | ItemEnum::Impl(_) => Visibility::Default,

        ItemEnum::Module(_)
        | ItemEnum::ExternCrate { .. }
        | ItemEnum::Use(_)
        | ItemEnum::Union(_)
        | ItemEnum::Struct(_)
        | ItemEnum::StructField(_)
        | ItemEnum::Enum(_)
        | ItemEnum::Function(_)
        | ItemEnum::Trait(_)
        | ItemEnum::TraitAlias(_)
        | ItemEnum::TypeAlias(_)
        | ItemEnum::Constant { .. }
        | ItemEnum::Static(_)
        | ItemEnum::ExternType
        | ItemEnum::Macro(_)
        | ItemEnum::ProcMacro(_)
        | ItemEnum::Primitive(_) => Visibility::Public,
    }
}
