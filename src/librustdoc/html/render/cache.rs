use std::collections::BTreeMap;
use std::path::Path;

use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::sym;
use serde::Serialize;

use crate::clean::types::GetDefId;
use crate::clean::{self, AttributesExt};
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::render::{plain_text_summary, shorten};
use crate::html::render::{Generic, IndexItem, IndexItemFunctionType, RenderType, TypeWithKind};

/// Indicates where an external crate can be found.
pub enum ExternalLocation {
    /// Remote URL root of the external crate
    Remote(String),
    /// This external crate can be found in the local doc/ folder
    Local,
    /// The external crate could not be found.
    Unknown,
}

/// Attempts to find where an external crate is located, given that we're
/// rendering in to the specified source destination.
pub fn extern_location(
    e: &clean::ExternalCrate,
    extern_url: Option<&str>,
    dst: &Path,
) -> ExternalLocation {
    use ExternalLocation::*;
    // See if there's documentation generated into the local directory
    let local_location = dst.join(&e.name);
    if local_location.is_dir() {
        return Local;
    }

    if let Some(url) = extern_url {
        let mut url = url.to_string();
        if !url.ends_with('/') {
            url.push('/');
        }
        return Remote(url);
    }

    // Failing that, see if there's an attribute specifying where to find this
    // external crate
    e.attrs
        .lists(sym::doc)
        .filter(|a| a.has_name(sym::html_root_url))
        .filter_map(|a| a.value_str())
        .map(|url| {
            let mut url = url.to_string();
            if !url.ends_with('/') {
                url.push('/')
            }
            Remote(url)
        })
        .next()
        .unwrap_or(Unknown) // Well, at least we tried.
}

/// Builds the search index from the collected metadata
pub fn build_index(krate: &clean::Crate, cache: &mut Cache) -> String {
    let mut defid_to_pathid = FxHashMap::default();
    let mut crate_items = Vec::with_capacity(cache.search_index.len());
    let mut crate_paths = vec![];

    let Cache { ref mut search_index, ref orphan_impl_items, ref paths, ref mut aliases, .. } =
        *cache;

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &(did, ref item) in orphan_impl_items {
        if let Some(&(ref fqp, _)) = paths.get(&did) {
            search_index.push(IndexItem {
                ty: item.type_(),
                name: item.name.clone().unwrap(),
                path: fqp[..fqp.len() - 1].join("::"),
                desc: shorten(plain_text_summary(item.doc_value())),
                parent: Some(did),
                parent_idx: None,
                search_type: get_index_search_type(&item),
            });
            for alias in item.attrs.get_doc_aliases() {
                aliases
                    .entry(alias.to_lowercase())
                    .or_insert(Vec::new())
                    .push(search_index.len() - 1);
            }
        }
    }

    // Reduce `DefId` in paths into smaller sequential numbers,
    // and prune the paths that do not appear in the index.
    let mut lastpath = String::new();
    let mut lastpathid = 0usize;

    for item in search_index {
        item.parent_idx = item.parent.and_then(|defid| {
            if defid_to_pathid.contains_key(&defid) {
                defid_to_pathid.get(&defid).copied()
            } else {
                let pathid = lastpathid;
                defid_to_pathid.insert(defid, pathid);
                lastpathid += 1;

                if let Some(&(ref fqp, short)) = paths.get(&defid) {
                    crate_paths.push((short, fqp.last().unwrap().clone()));
                    Some(pathid)
                } else {
                    None
                }
            }
        });

        // Omit the parent path if it is same to that of the prior item.
        if lastpath == item.path {
            item.path.clear();
        } else {
            lastpath = item.path.clone();
        }
        crate_items.push(&*item);
    }

    let crate_doc = krate
        .module
        .as_ref()
        .map(|module| shorten(plain_text_summary(module.doc_value())))
        .unwrap_or_default();

    #[derive(Serialize)]
    struct CrateData<'a> {
        doc: String,
        #[serde(rename = "i")]
        items: Vec<&'a IndexItem>,
        #[serde(rename = "p")]
        paths: Vec<(ItemType, String)>,
        // The String is alias name and the vec is the list of the elements with this alias.
        //
        // To be noted: the `usize` elements are indexes to `items`.
        #[serde(rename = "a")]
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        aliases: &'a BTreeMap<String, Vec<usize>>,
    }

    // Collect the index into a string
    format!(
        r#""{}":{}"#,
        krate.name,
        serde_json::to_string(&CrateData {
            doc: crate_doc,
            items: crate_items,
            paths: crate_paths,
            aliases,
        })
        .expect("failed serde conversion")
        // All these `replace` calls are because we have to go through JS string for JSON content.
        .replace(r"\", r"\\")
        .replace("'", r"\'")
        // We need to escape double quotes for the JSON.
        .replace("\\\"", "\\\\\"")
    )
}

crate fn get_index_search_type(item: &clean::Item) -> Option<IndexItemFunctionType> {
    let (all_types, ret_types) = match item.kind {
        clean::FunctionItem(ref f) => (&f.all_types, &f.ret_types),
        clean::MethodItem(ref m) => (&m.all_types, &m.ret_types),
        clean::TyMethodItem(ref m) => (&m.all_types, &m.ret_types),
        _ => return None,
    };

    let inputs = all_types
        .iter()
        .map(|(ty, kind)| TypeWithKind::from((get_index_type(&ty), *kind)))
        .filter(|a| a.ty.name.is_some())
        .collect();
    let output = ret_types
        .iter()
        .map(|(ty, kind)| TypeWithKind::from((get_index_type(&ty), *kind)))
        .filter(|a| a.ty.name.is_some())
        .collect::<Vec<_>>();
    let output = if output.is_empty() { None } else { Some(output) };

    Some(IndexItemFunctionType { inputs, output })
}

fn get_index_type(clean_type: &clean::Type) -> RenderType {
    RenderType {
        ty: clean_type.def_id(),
        idx: None,
        name: get_index_type_name(clean_type, true).map(|s| s.to_ascii_lowercase()),
        generics: get_generics(clean_type),
    }
}

fn get_index_type_name(clean_type: &clean::Type, accept_generic: bool) -> Option<String> {
    match *clean_type {
        clean::ResolvedPath { ref path, .. } => {
            let segments = &path.segments;
            let path_segment = segments.iter().last().unwrap_or_else(|| {
                panic!(
                "get_index_type_name(clean_type: {:?}, accept_generic: {:?}) had length zero path",
                clean_type, accept_generic
            )
            });
            Some(path_segment.name.clone())
        }
        clean::Generic(ref s) if accept_generic => Some(s.clone()),
        clean::Primitive(ref p) => Some(format!("{:?}", p)),
        clean::BorrowedRef { ref type_, .. } => get_index_type_name(type_, accept_generic),
        // FIXME: add all from clean::Type.
        _ => None,
    }
}

fn get_generics(clean_type: &clean::Type) -> Option<Vec<Generic>> {
    clean_type.generics().and_then(|types| {
        let r = types
            .iter()
            .filter_map(|t| {
                get_index_type_name(t, false).map(|name| Generic {
                    name: name.to_ascii_lowercase(),
                    defid: t.def_id(),
                    idx: None,
                })
            })
            .collect::<Vec<_>>();
        if r.is_empty() { None } else { Some(r) }
    })
}
