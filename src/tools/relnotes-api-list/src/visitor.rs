use crate::store::{Store, StoreCrateId, StoreItem};
use anyhow::{Context, Error, bail};
use rustdoc_json_types::{Id, ItemEnum, StructKind};
use std::iter::once;

pub(crate) trait Visitor<'a> {
    type Result: MergeResults;

    fn store(&self) -> &'a Store;
    fn visit_item(&mut self, item: &StoreItem<'a>) -> Result<Self::Result, Error>;
}

pub(crate) fn walk_item<'a, V: Visitor<'a>>(
    v: &mut V,
    item: &StoreItem<'a>,
) -> Result<V::Result, Error> {
    let store = v.store();
    let krate = item.crate_id;
    match &item.inner {
        ItemEnum::Module(module) => walk_ids(krate, &mut module.items.iter(), v),
        ItemEnum::Impl(impl_) => walk_ids(krate, &mut impl_.items.iter(), v),
        ItemEnum::Primitive(primitive) => walk_ids(krate, &mut primitive.impls.iter(), v),
        ItemEnum::Union(union) => {
            walk_ids(krate, &mut union.fields.iter().chain(union.impls.iter()), v)
        }
        ItemEnum::Enum(enum_) => {
            walk_ids(krate, &mut enum_.variants.iter().chain(enum_.impls.iter()), v)
        }
        ItemEnum::Trait(trait_) => {
            // This intentionally doesn't walk through trait_.implementations, as we don't care
            // about those when generating the API list.
            walk_ids(krate, &mut trait_.items.iter(), v)
        }
        ItemEnum::Struct(struct_) => match &struct_.kind {
            StructKind::Unit => walk_ids(krate, &mut struct_.impls.iter(), v),
            StructKind::Tuple(ids) => walk_ids(
                krate,
                &mut struct_.impls.iter().chain(ids.iter().filter_map(|f| f.as_ref())),
                v,
            ),
            StructKind::Plain { fields, .. } => {
                walk_ids(krate, &mut struct_.impls.iter().chain(fields.iter()), v)
            }
        },

        ItemEnum::Use(use_) => {
            if let Some(used_id) = use_.id {
                if use_.source.contains("ParseFloatError") {
                    return Ok(V::Result::default()); // TODO: groan
                }
                if use_.source == "quote::quote" {
                    return Ok(V::Result::default()); // TODO: groan
                }
                if use_.source == "quote::quote_span" {
                    return Ok(V::Result::default()); // TODO: groan
                }

                let dest =
                    store.resolve_use(krate, used_id).context("could not resolve the use")?;

                if use_.is_glob {
                    match &store.item(dest.krate, dest.item)?.inner {
                        ItemEnum::Module(m) => walk_ids(dest.krate, &mut m.items.iter(), v),
                        ItemEnum::Enum(e) => walk_ids(dest.krate, &mut e.variants.iter(), v),
                        _ => bail!("glob use doesn't point to a module or enum"),
                    }
                } else {
                    walk_ids(dest.krate, &mut once(&dest.item), v)
                }
            } else {
                // Do not deal with re-exports of primitives (which have no ID in their use).
                Ok(V::Result::default())
            }
        }

        ItemEnum::AssocConst { .. }
        | ItemEnum::AssocType { .. }
        | ItemEnum::ProcMacro(_)
        | ItemEnum::Macro(_)
        | ItemEnum::ExternType
        | ItemEnum::Static(_)
        | ItemEnum::Constant { .. }
        | ItemEnum::TypeAlias(_)
        | ItemEnum::TraitAlias(_)
        | ItemEnum::ExternCrate { .. }
        | ItemEnum::StructField(_)
        | ItemEnum::Variant(_)
        | ItemEnum::Function(_) => Ok(V::Result::default()),
    }
}

fn walk_ids<'a, 'b, V: Visitor<'a>>(
    crate_id: StoreCrateId,
    ids: &mut dyn Iterator<Item = &'b Id>,
    visitor: &mut V,
) -> Result<V::Result, Error> {
    let store = visitor.store();
    let mut result = V::Result::default();
    for id in ids {
        let item = store.item(crate_id, *id)?;
        match visitor.visit_item(&item) {
            Ok(new) => result.merge_other(new),
            Err(err) => {
                let name = match &item.name {
                    Some(name) => format!("with name {name}"),
                    None => format!("with ID {id:?}"),
                };
                return Err(err.context(format!("while visiting item {name}")));
            }
        }
    }
    Ok(result)
}

pub(crate) trait MergeResults: Default {
    fn merge_other(&mut self, other: Self);
}

impl MergeResults for () {
    fn merge_other(&mut self, _other: Self) {}
}

impl<T> MergeResults for Vec<T> {
    fn merge_other(&mut self, other: Self) {
        self.extend(other)
    }
}
