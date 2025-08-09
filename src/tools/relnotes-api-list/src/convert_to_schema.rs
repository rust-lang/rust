use std::collections::HashSet;

use anyhow::{Context, Error, bail};
use rustdoc_json_types::{Id, ItemEnum, MacroKind, Visibility};

use crate::PUBLIC_CRATES;
use crate::pretty_print::PrettyPrinter;
use crate::schema::SchemaItem;
use crate::stability::{Stability, StabilityStore};
use crate::store::{Store, StoreCrateId, StoreItem};
use crate::visitor::{Visitor, walk_item};

pub(crate) struct ConvertToSchema<'a> {
    store: &'a Store,
    stability: &'a StabilityStore,

    within_impl: bool,
    within_trait: bool,
    url_stack: Vec<String>,
    name_stack: Vec<String>,
    reachable_without_use: HashSet<(StoreCrateId, Id)>,
}

impl<'a> Visitor<'a> for ConvertToSchema<'a> {
    type Result = Vec<SchemaItem>;

    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<Vec<SchemaItem>, Error> {
        if item.visibility != self.expected_visibility(item) {
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

        let mut pop_url = false;
        match self.url_fragment(item)? {
            UrlChunk::Directory(dir) => {
                pop_url = true;
                self.url_stack.push(format!("{dir}/"));
            }
            UrlChunk::Item(kind) => {
                pop_url = true;
                let name = item.require_name()?;
                if self.url_stack.last().map(|last| last.ends_with(".html")).unwrap_or(false) {
                    self.url_stack.push(format!("#{kind}.{name}"))
                } else {
                    self.url_stack.push(format!("{kind}.{name}.html"));
                }
            }
            UrlChunk::None => {}
        }

        let mut restore_within_impl = None;
        let mut restore_within_trait = None;
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
            | ItemEnum::TypeAlias(_)
            | ItemEnum::Union(_)
            | ItemEnum::Variant(_)
            | ItemEnum::ProcMacro(_)
            | ItemEnum::Function(_) => self.include(item),

            ItemEnum::Trait(_) => {
                restore_within_trait = Some(self.within_trait);
                self.within_trait = true;

                self.include(item)
            }

            ItemEnum::Primitive(p) => {
                // We don't want primitives to have the `std::` prefix.
                let old_stack = std::mem::replace(&mut self.name_stack, vec![p.name.clone()]);
                let result = self.include(item);
                self.name_stack = old_stack;
                result
            }

            ItemEnum::Use(_) => {
                if self.should_inline_use(item)? {
                    walk_item(self, item)
                } else {
                    // TODO: do we want to include an item for public re-exports? Probably yes!
                    Ok(Vec::new())
                }
            }

            ItemEnum::Impl(impl_) => {
                if impl_.trait_.is_some() {
                    let url = format!(
                        "{}#{}",
                        self.current_url(),
                        &PrettyPrinter::for_url(self.store, item.crate_id)
                            .pretty_impl(impl_)
                            .context("failed to generate the URL of the impl")?
                            .as_urlencoded()
                    );
                    if url.contains("crate::TokenStream") {
                        //bail!("{item:#?}");
                    }
                    Ok(vec![SchemaItem {
                        name: PrettyPrinter::for_name(self.store, item.crate_id)
                            .pretty_impl(impl_)
                            .context("failed to generate the name of the impl")?
                            .as_text(),
                        deprecated: item.deprecation.is_some(),
                        url: Some(url),

                        // We are intentionally not walking inside impls of traits: we don't want
                        // all types in the standard library to show up in the changelog if a new
                        // item is added in a trait.
                        children: Vec::new(),
                    }])
                } else {
                    restore_within_impl = Some(self.within_impl);
                    self.within_impl = true;

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
        if pop_url {
            self.url_stack.pop();
        }
        if let Some(restore) = restore_within_impl {
            self.within_impl = restore;
        }
        if let Some(restore) = restore_within_trait {
            self.within_trait = restore;
        }
        result
    }

    fn store(&self) -> &'a Store {
        self.store
    }
}

impl<'a> ConvertToSchema<'a> {
    pub(crate) fn new(store: &'a Store, stability: &'a StabilityStore) -> Result<Self, Error> {
        Ok(Self {
            store,
            stability,
            within_impl: false,
            within_trait: false,
            name_stack: Vec::new(),
            url_stack: Vec::new(),
            reachable_without_use: reachable_without_use(store)?,
        })
    }

    fn include(&mut self, item: &StoreItem<'a>) -> Result<Vec<SchemaItem>, Error> {
        let item = SchemaItem {
            name: self.name_stack.join("::"),
            url: Some(self.current_url()),
            deprecated: item.deprecation.is_some(),
            children: walk_item(self, item)?,
        };
        Ok(vec![item])
    }

    fn should_inline_use(&self, use_: &StoreItem<'a>) -> Result<bool, Error> {
        for attr in &use_.attrs {
            if attr == "#[doc(no_inline)]" {
                return Ok(false);
            } else if attr == "#[doc(inline)]" {
                return Ok(true);
            }
        }

        let resolved = self.store.resolve_use_recursive(use_.crate_id, use_.id)?;
        Ok(!self.reachable_without_use.contains(&(resolved.krate, resolved.item)))
    }

    fn current_url(&self) -> String {
        let mut url = self.url_stack.join("");
        if url.ends_with('/') {
            url.push_str("index.html");
        }
        url
    }

    fn url_fragment(&self, item: &StoreItem<'a>) -> Result<UrlChunk<'a>, Error> {
        Ok(UrlChunk::Item(match &item.inner {
            // The fragment returned here will be used here as `{fragment}.{name}.html` or
            // `#{fragment}.{name}`, depending on whether the item is nested in another item.
            ItemEnum::Union(_) => "union",
            ItemEnum::Struct(_) => "struct",
            ItemEnum::StructField(_) => "structfield",
            ItemEnum::Enum(_) => "enum",
            ItemEnum::Variant(_) => "variant",
            ItemEnum::Function(f) if self.within_trait && !f.has_body => "tymethod",
            ItemEnum::Function(_) if self.within_impl || self.within_trait => "method",
            ItemEnum::Function(_) => "fn",
            ItemEnum::Trait(_) => "trait",
            ItemEnum::Constant { .. } => "constant",
            ItemEnum::Static(_) => "static",
            ItemEnum::Macro(_) => "macro",
            ItemEnum::Primitive(_) => "primitive",
            ItemEnum::AssocConst { .. } => "associatedconstant",
            ItemEnum::AssocType { .. } => "associatedtype",
            ItemEnum::ProcMacro(proc_macro) => match proc_macro.kind {
                MacroKind::Bang => "macro",
                MacroKind::Attr => "attr",
                MacroKind::Derive => "derive",
            },

            ItemEnum::Module(_) => return Ok(UrlChunk::Directory(item.require_name()?)),

            ItemEnum::ExternType
            | ItemEnum::ExternCrate { .. }
            | ItemEnum::Impl(_)
            | ItemEnum::TraitAlias(_)
            | ItemEnum::TypeAlias(_)
            | ItemEnum::Use(_) => return Ok(UrlChunk::None),
        }))
    }

    /// Some items don't have a visibility associated to them, and are instead public by default. This
    /// function determines what visibility a public item must have.
    fn expected_visibility(&self, item: &StoreItem<'_>) -> Visibility {
        if self.within_trait {
            return Visibility::Default;
        }

        match &item.inner {
            ItemEnum::Variant(_) | ItemEnum::Impl(_) => Visibility::Default,

            ItemEnum::AssocType { .. }
            | ItemEnum::AssocConst { .. }
            | ItemEnum::Module(_)
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
}

enum UrlChunk<'a> {
    Directory(&'a str),
    Item(&'static str),
    None,
}

/// Unless overridden with `#[doc(inline)]` or `#[doc(no_inline)]`, rustdoc chooses whether to
/// inline an `use` or not by checking if it's otherwise reachable without `use`s. For example:
///
/// * `core::prelude::v1::Iterator` should not be inlined, since it's an `use` pointing to
///   `core::iter::Iterator`, which is accessible directly.
///
/// * `core::mem::MaybeUninit` should be inlined, since it's an `use` pointing to the *private*
///   `core::mem::maybe_uninit::MaybeUninit`.
///
/// This function walks the JSON to collect all items that are reachable without an use.
fn reachable_without_use(store: &Store) -> Result<HashSet<(StoreCrateId, Id)>, Error> {
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
            // Don't walk into `use`s.
            if let ItemEnum::Use(_) = &item.inner {
                return Ok(());
            }

            self.reachable.insert((item.crate_id, item.id));
            walk_item(self, item)
        }
    }

    let mut visitor = ReachableWithoutUse { store, reachable: HashSet::new() };
    for crate_id in store.crate_ids() {
        // Items defined in non-public dependencies (for example, as of 2025, the std_detect crate),
        // are never reachable in the documentation without a `use`, since we don't generate docs
        // for those crates. So, only look in publicly accessible crates.
        if PUBLIC_CRATES.contains(&store.crate_name(crate_id)) {
            visitor.visit_item(&store.crate_root(crate_id)?).with_context(|| {
                format!(
                    "failed to check the reachability of items in crate {}",
                    store.crate_name(crate_id)
                )
            })?;
        }
    }
    Ok(visitor.reachable)
}
