pub(crate) mod encode;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, VecDeque};

use encode::{bitmap_to_string, write_vlqhex_to_string};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::sym;
use rustc_span::symbol::{Symbol, kw};
use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};
use thin_vec::ThinVec;
use tracing::instrument;

use crate::clean::types::{Function, Generics, ItemId, Type, WherePredicate};
use crate::clean::{self, utils};
use crate::formats::cache::{Cache, OrphanImplItem};
use crate::formats::item_type::ItemType;
use crate::html::format::join_with_double_colon;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::ordered_json::OrderedJson;
use crate::html::render::{self, IndexItem, IndexItemFunctionType, RenderType, RenderTypeId};

/// The serialized search description sharded version
///
/// The `index` is a JSON-encoded list of names and other information.
///
/// The desc has newlined descriptions, split up by size into 128KiB shards.
/// For example, `(4, "foo\nbar\nbaz\nquux")`.
///
/// There is no single, optimal size for these shards, because it depends on
/// configuration values that we can't predict or control, such as the version
/// of HTTP used (HTTP/1.1 would work better with larger files, while HTTP/2
/// and 3 are more agnostic), transport compression (gzip, zstd, etc), whether
/// the search query is going to produce a large number of results or a small
/// number, the bandwidth delay product of the network...
///
/// Gzipping some standard library descriptions to guess what transport
/// compression will do, the compressed file sizes can be as small as 4.9KiB
/// or as large as 18KiB (ignoring the final 1.9KiB shard of leftovers).
/// A "reasonable" range for files is for them to be bigger than 1KiB,
/// since that's about the amount of data that can be transferred in a
/// single TCP packet, and 64KiB, the maximum amount of data that
/// TCP can transfer in a single round trip without extensions.
///
/// [1]: https://en.wikipedia.org/wiki/Maximum_transmission_unit#MTUs_for_common_media
/// [2]: https://en.wikipedia.org/wiki/Sliding_window_protocol#Basic_concept
/// [3]: https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/description-tcp-features
pub(crate) struct SerializedSearchIndex {
    pub(crate) index: OrderedJson,
    pub(crate) desc: Vec<(usize, String)>,
}

const DESC_INDEX_SHARD_LEN: usize = 128 * 1024;

/// Builds the search index from the collected metadata
pub(crate) fn build_index(
    krate: &clean::Crate,
    cache: &mut Cache,
    tcx: TyCtxt<'_>,
) -> SerializedSearchIndex {
    // Maps from ID to position in the `crate_paths` array.
    let mut itemid_to_pathid = FxHashMap::default();
    let mut primitives = FxHashMap::default();
    let mut associated_types = FxHashMap::default();

    // item type, display path, re-exported internal path
    let mut crate_paths: Vec<(ItemType, Vec<Symbol>, Option<Vec<Symbol>>, bool)> = vec![];

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &OrphanImplItem { impl_id, parent, ref item, ref impl_generics } in &cache.orphan_impl_items
    {
        if let Some((fqp, _)) = cache.paths.get(&parent) {
            let desc = short_markdown_summary(&item.doc_value(), &item.link_names(cache));
            cache.search_index.push(IndexItem {
                ty: item.type_(),
                defid: item.item_id.as_def_id(),
                name: item.name.unwrap(),
                path: join_with_double_colon(&fqp[..fqp.len() - 1]),
                desc,
                parent: Some(parent),
                parent_idx: None,
                exact_path: None,
                impl_id,
                search_type: get_function_type_for_search(
                    item,
                    tcx,
                    impl_generics.as_ref(),
                    Some(parent),
                    cache,
                ),
                aliases: item.attrs.get_doc_aliases(),
                deprecation: item.deprecation(tcx),
            });
        }
    }

    let crate_doc =
        short_markdown_summary(&krate.module.doc_value(), &krate.module.link_names(cache));

    // Aliases added through `#[doc(alias = "...")]`. Since a few items can have the same alias,
    // we need the alias element to have an array of items.
    let mut aliases: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    // Sort search index items. This improves the compressibility of the search index.
    cache.search_index.sort_unstable_by(|k1, k2| {
        // `sort_unstable_by_key` produces lifetime errors
        // HACK(rustdoc): should not be sorting `CrateNum` or `DefIndex`, this will soon go away, too
        let k1 = (&k1.path, k1.name.as_str(), &k1.ty, k1.parent.map(|id| (id.index, id.krate)));
        let k2 = (&k2.path, k2.name.as_str(), &k2.ty, k2.parent.map(|id| (id.index, id.krate)));
        Ord::cmp(&k1, &k2)
    });

    // Set up alias indexes.
    for (i, item) in cache.search_index.iter().enumerate() {
        for alias in &item.aliases[..] {
            aliases.entry(alias.as_str().to_lowercase()).or_default().push(i);
        }
    }

    // Reduce `DefId` in paths into smaller sequential numbers,
    // and prune the paths that do not appear in the index.
    let mut lastpath = "";
    let mut lastpathid = 0isize;

    // First, on function signatures
    let mut search_index = std::mem::take(&mut cache.search_index);
    for item in search_index.iter_mut() {
        fn insert_into_map<F: std::hash::Hash + Eq>(
            map: &mut FxHashMap<F, isize>,
            itemid: F,
            lastpathid: &mut isize,
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>, Option<Vec<Symbol>>, bool)>,
            item_type: ItemType,
            path: &[Symbol],
            exact_path: Option<&[Symbol]>,
            search_unbox: bool,
        ) -> RenderTypeId {
            match map.entry(itemid) {
                Entry::Occupied(entry) => RenderTypeId::Index(*entry.get()),
                Entry::Vacant(entry) => {
                    let pathid = *lastpathid;
                    entry.insert(pathid);
                    *lastpathid += 1;
                    crate_paths.push((
                        item_type,
                        path.to_vec(),
                        exact_path.map(|path| path.to_vec()),
                        search_unbox,
                    ));
                    RenderTypeId::Index(pathid)
                }
            }
        }

        fn convert_render_type_id(
            id: RenderTypeId,
            cache: &mut Cache,
            itemid_to_pathid: &mut FxHashMap<ItemId, isize>,
            primitives: &mut FxHashMap<Symbol, isize>,
            associated_types: &mut FxHashMap<Symbol, isize>,
            lastpathid: &mut isize,
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>, Option<Vec<Symbol>>, bool)>,
            tcx: TyCtxt<'_>,
        ) -> Option<RenderTypeId> {
            use crate::clean::PrimitiveType;
            let Cache { ref paths, ref external_paths, ref exact_paths, .. } = *cache;
            let search_unbox = match id {
                RenderTypeId::Mut => false,
                RenderTypeId::DefId(defid) => utils::has_doc_flag(tcx, defid, sym::search_unbox),
                RenderTypeId::Primitive(PrimitiveType::Reference | PrimitiveType::Tuple) => true,
                RenderTypeId::Primitive(..) => false,
                RenderTypeId::AssociatedType(..) => false,
                // this bool is only used by `insert_into_map`, so it doesn't matter what we set here
                // because Index means we've already inserted into the map
                RenderTypeId::Index(_) => false,
            };
            match id {
                RenderTypeId::Mut => Some(insert_into_map(
                    primitives,
                    kw::Mut,
                    lastpathid,
                    crate_paths,
                    ItemType::Keyword,
                    &[kw::Mut],
                    None,
                    search_unbox,
                )),
                RenderTypeId::DefId(defid) => {
                    if let Some(&(ref fqp, item_type)) =
                        paths.get(&defid).or_else(|| external_paths.get(&defid))
                    {
                        let exact_fqp = exact_paths
                            .get(&defid)
                            .or_else(|| external_paths.get(&defid).map(|(fqp, _)| fqp))
                            // Re-exports only count if the name is exactly the same.
                            // This is a size optimization, since it means we only need
                            // to store the name once (and the path is re-used for everything
                            // exported from this same module). It's also likely to Do
                            // What I Mean, since if a re-export changes the name, it might
                            // also be a change in semantic meaning.
                            .filter(|this_fqp| this_fqp.last() == fqp.last());
                        Some(insert_into_map(
                            itemid_to_pathid,
                            ItemId::DefId(defid),
                            lastpathid,
                            crate_paths,
                            item_type,
                            fqp,
                            exact_fqp.map(|x| &x[..]).filter(|exact_fqp| exact_fqp != fqp),
                            search_unbox,
                        ))
                    } else {
                        None
                    }
                }
                RenderTypeId::Primitive(primitive) => {
                    let sym = primitive.as_sym();
                    Some(insert_into_map(
                        primitives,
                        sym,
                        lastpathid,
                        crate_paths,
                        ItemType::Primitive,
                        &[sym],
                        None,
                        search_unbox,
                    ))
                }
                RenderTypeId::Index(_) => Some(id),
                RenderTypeId::AssociatedType(sym) => Some(insert_into_map(
                    associated_types,
                    sym,
                    lastpathid,
                    crate_paths,
                    ItemType::AssocType,
                    &[sym],
                    None,
                    search_unbox,
                )),
            }
        }

        fn convert_render_type(
            ty: &mut RenderType,
            cache: &mut Cache,
            itemid_to_pathid: &mut FxHashMap<ItemId, isize>,
            primitives: &mut FxHashMap<Symbol, isize>,
            associated_types: &mut FxHashMap<Symbol, isize>,
            lastpathid: &mut isize,
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>, Option<Vec<Symbol>>, bool)>,
            tcx: TyCtxt<'_>,
        ) {
            if let Some(generics) = &mut ty.generics {
                for item in generics {
                    convert_render_type(
                        item,
                        cache,
                        itemid_to_pathid,
                        primitives,
                        associated_types,
                        lastpathid,
                        crate_paths,
                        tcx,
                    );
                }
            }
            if let Some(bindings) = &mut ty.bindings {
                bindings.retain_mut(|(associated_type, constraints)| {
                    let converted_associated_type = convert_render_type_id(
                        *associated_type,
                        cache,
                        itemid_to_pathid,
                        primitives,
                        associated_types,
                        lastpathid,
                        crate_paths,
                        tcx,
                    );
                    let Some(converted_associated_type) = converted_associated_type else {
                        return false;
                    };
                    *associated_type = converted_associated_type;
                    for constraint in constraints {
                        convert_render_type(
                            constraint,
                            cache,
                            itemid_to_pathid,
                            primitives,
                            associated_types,
                            lastpathid,
                            crate_paths,
                            tcx,
                        );
                    }
                    true
                });
            }
            let Some(id) = ty.id else {
                assert!(ty.generics.is_some());
                return;
            };
            ty.id = convert_render_type_id(
                id,
                cache,
                itemid_to_pathid,
                primitives,
                associated_types,
                lastpathid,
                crate_paths,
                tcx,
            );
        }
        if let Some(search_type) = &mut item.search_type {
            for item in &mut search_type.inputs {
                convert_render_type(
                    item,
                    cache,
                    &mut itemid_to_pathid,
                    &mut primitives,
                    &mut associated_types,
                    &mut lastpathid,
                    &mut crate_paths,
                    tcx,
                );
            }
            for item in &mut search_type.output {
                convert_render_type(
                    item,
                    cache,
                    &mut itemid_to_pathid,
                    &mut primitives,
                    &mut associated_types,
                    &mut lastpathid,
                    &mut crate_paths,
                    tcx,
                );
            }
            for constraint in &mut search_type.where_clause {
                for trait_ in &mut constraint[..] {
                    convert_render_type(
                        trait_,
                        cache,
                        &mut itemid_to_pathid,
                        &mut primitives,
                        &mut associated_types,
                        &mut lastpathid,
                        &mut crate_paths,
                        tcx,
                    );
                }
            }
        }
    }

    let Cache { ref paths, ref exact_paths, ref external_paths, .. } = *cache;

    // Then, on parent modules
    let crate_items: Vec<&IndexItem> = search_index
        .iter_mut()
        .map(|item| {
            item.parent_idx =
                item.parent.and_then(|defid| match itemid_to_pathid.entry(ItemId::DefId(defid)) {
                    Entry::Occupied(entry) => Some(*entry.get()),
                    Entry::Vacant(entry) => {
                        let pathid = lastpathid;
                        entry.insert(pathid);
                        lastpathid += 1;

                        if let Some(&(ref fqp, short)) = paths.get(&defid) {
                            let exact_fqp = exact_paths
                                .get(&defid)
                                .or_else(|| external_paths.get(&defid).map(|(fqp, _)| fqp))
                                .filter(|exact_fqp| {
                                    exact_fqp.last() == Some(&item.name) && *exact_fqp != fqp
                                });
                            crate_paths.push((
                                short,
                                fqp.clone(),
                                exact_fqp.cloned(),
                                utils::has_doc_flag(tcx, defid, sym::search_unbox),
                            ));
                            Some(pathid)
                        } else {
                            None
                        }
                    }
                });

            if let Some(defid) = item.defid
                && item.parent_idx.is_none()
            {
                // If this is a re-export, retain the original path.
                // Associated items don't use this.
                // Their parent carries the exact fqp instead.
                let exact_fqp = exact_paths
                    .get(&defid)
                    .or_else(|| external_paths.get(&defid).map(|(fqp, _)| fqp));
                item.exact_path = exact_fqp.and_then(|fqp| {
                    // Re-exports only count if the name is exactly the same.
                    // This is a size optimization, since it means we only need
                    // to store the name once (and the path is re-used for everything
                    // exported from this same module). It's also likely to Do
                    // What I Mean, since if a re-export changes the name, it might
                    // also be a change in semantic meaning.
                    if fqp.last() != Some(&item.name) {
                        return None;
                    }
                    let path =
                        if item.ty == ItemType::Macro && tcx.has_attr(defid, sym::macro_export) {
                            // `#[macro_export]` always exports to the crate root.
                            tcx.crate_name(defid.krate).to_string()
                        } else {
                            if fqp.len() < 2 {
                                return None;
                            }
                            join_with_double_colon(&fqp[..fqp.len() - 1])
                        };
                    if path == item.path {
                        return None;
                    }
                    Some(path)
                });
            } else if let Some(parent_idx) = item.parent_idx {
                let i = <isize as TryInto<usize>>::try_into(parent_idx).unwrap();
                item.path = {
                    let p = &crate_paths[i].1;
                    join_with_double_colon(&p[..p.len() - 1])
                };
                item.exact_path =
                    crate_paths[i].2.as_ref().map(|xp| join_with_double_colon(&xp[..xp.len() - 1]));
            }

            // Omit the parent path if it is same to that of the prior item.
            if lastpath == item.path {
                item.path.clear();
            } else {
                lastpath = &item.path;
            }

            &*item
        })
        .collect();

    // Find associated items that need disambiguators
    let mut associated_item_duplicates = FxHashMap::<(isize, ItemType, Symbol), usize>::default();

    for &item in &crate_items {
        if item.impl_id.is_some()
            && let Some(parent_idx) = item.parent_idx
        {
            let count =
                associated_item_duplicates.entry((parent_idx, item.ty, item.name)).or_insert(0);
            *count += 1;
        }
    }

    let associated_item_disambiguators = crate_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let impl_id = ItemId::DefId(item.impl_id?);
            let parent_idx = item.parent_idx?;
            let count = *associated_item_duplicates.get(&(parent_idx, item.ty, item.name))?;
            if count > 1 { Some((index, render::get_id_for_impl(tcx, impl_id))) } else { None }
        })
        .collect::<Vec<_>>();

    struct CrateData<'a> {
        items: Vec<&'a IndexItem>,
        paths: Vec<(ItemType, Vec<Symbol>, Option<Vec<Symbol>>, bool)>,
        // The String is alias name and the vec is the list of the elements with this alias.
        //
        // To be noted: the `usize` elements are indexes to `items`.
        aliases: &'a BTreeMap<String, Vec<usize>>,
        // Used when a type has more than one impl with an associated item with the same name.
        associated_item_disambiguators: &'a Vec<(usize, String)>,
        // A list of shard lengths encoded as vlqhex. See the comment in write_vlqhex_to_string
        // for information on the format.
        desc_index: String,
        // A list of items with no description. This is eventually turned into a bitmap.
        empty_desc: Vec<u32>,
    }

    struct Paths {
        ty: ItemType,
        name: Symbol,
        path: Option<usize>,
        exact_path: Option<usize>,
        search_unbox: bool,
    }

    impl Serialize for Paths {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut seq = serializer.serialize_seq(None)?;
            seq.serialize_element(&self.ty)?;
            seq.serialize_element(self.name.as_str())?;
            if let Some(ref path) = self.path {
                seq.serialize_element(path)?;
            }
            if let Some(ref path) = self.exact_path {
                assert!(self.path.is_some());
                seq.serialize_element(path)?;
            }
            if self.search_unbox {
                if self.path.is_none() {
                    seq.serialize_element(&None::<u8>)?;
                }
                if self.exact_path.is_none() {
                    seq.serialize_element(&None::<u8>)?;
                }
                seq.serialize_element(&1)?;
            }
            seq.end()
        }
    }

    impl Serialize for CrateData<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut extra_paths = FxHashMap::default();
            // We need to keep the order of insertion, hence why we use an `IndexMap`. Then we will
            // insert these "extra paths" (which are paths of items from external crates) into the
            // `full_paths` list at the end.
            let mut revert_extra_paths = FxIndexMap::default();
            let mut mod_paths = FxHashMap::default();
            for (index, item) in self.items.iter().enumerate() {
                if item.path.is_empty() {
                    continue;
                }
                mod_paths.insert(&item.path, index);
            }
            let mut paths = Vec::with_capacity(self.paths.len());
            for &(ty, ref path, ref exact, search_unbox) in &self.paths {
                if path.len() < 2 {
                    paths.push(Paths {
                        ty,
                        name: path[0],
                        path: None,
                        exact_path: None,
                        search_unbox,
                    });
                    continue;
                }
                let full_path = join_with_double_colon(&path[..path.len() - 1]);
                let full_exact_path = exact
                    .as_ref()
                    .filter(|exact| exact.last() == path.last() && exact.len() >= 2)
                    .map(|exact| join_with_double_colon(&exact[..exact.len() - 1]));
                let exact_path = extra_paths.len() + self.items.len();
                let exact_path = full_exact_path.as_ref().map(|full_exact_path| match extra_paths
                    .entry(full_exact_path.clone())
                {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        if let Some(index) = mod_paths.get(&full_exact_path) {
                            return *index;
                        }
                        entry.insert(exact_path);
                        if !revert_extra_paths.contains_key(&exact_path) {
                            revert_extra_paths.insert(exact_path, full_exact_path.clone());
                        }
                        exact_path
                    }
                });
                if let Some(index) = mod_paths.get(&full_path) {
                    paths.push(Paths {
                        ty,
                        name: *path.last().unwrap(),
                        path: Some(*index),
                        exact_path,
                        search_unbox,
                    });
                    continue;
                }
                // It means it comes from an external crate so the item and its path will be
                // stored into another array.
                //
                // `index` is put after the last `mod_paths`
                let index = extra_paths.len() + self.items.len();
                match extra_paths.entry(full_path.clone()) {
                    Entry::Occupied(entry) => {
                        paths.push(Paths {
                            ty,
                            name: *path.last().unwrap(),
                            path: Some(*entry.get()),
                            exact_path,
                            search_unbox,
                        });
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(index);
                        if !revert_extra_paths.contains_key(&index) {
                            revert_extra_paths.insert(index, full_path);
                        }
                        paths.push(Paths {
                            ty,
                            name: *path.last().unwrap(),
                            path: Some(index),
                            exact_path,
                            search_unbox,
                        });
                    }
                }
            }

            // Direct exports use adjacent arrays for the current crate's items,
            // but re-exported exact paths don't.
            let mut re_exports = Vec::new();
            for (item_index, item) in self.items.iter().enumerate() {
                if let Some(exact_path) = item.exact_path.as_ref() {
                    if let Some(path_index) = mod_paths.get(&exact_path) {
                        re_exports.push((item_index, *path_index));
                    } else {
                        let path_index = extra_paths.len() + self.items.len();
                        let path_index = match extra_paths.entry(exact_path.clone()) {
                            Entry::Occupied(entry) => *entry.get(),
                            Entry::Vacant(entry) => {
                                entry.insert(path_index);
                                if !revert_extra_paths.contains_key(&path_index) {
                                    revert_extra_paths.insert(path_index, exact_path.clone());
                                }
                                path_index
                            }
                        };
                        re_exports.push((item_index, path_index));
                    }
                }
            }

            let mut names = Vec::with_capacity(self.items.len());
            let mut types = String::with_capacity(self.items.len());
            let mut full_paths = Vec::with_capacity(self.items.len());
            let mut parents = String::with_capacity(self.items.len());
            let mut parents_backref_queue = VecDeque::new();
            let mut functions = String::with_capacity(self.items.len());
            let mut deprecated = Vec::with_capacity(self.items.len());

            let mut type_backref_queue = VecDeque::new();

            let mut last_name = None;
            for (index, item) in self.items.iter().enumerate() {
                let n = item.ty as u8;
                let c = char::from(n + b'A');
                assert!(c <= 'z', "item types must fit within ASCII printables");
                types.push(c);

                assert_eq!(
                    item.parent.is_some(),
                    item.parent_idx.is_some(),
                    "`{}` is missing idx",
                    item.name
                );
                assert!(
                    parents_backref_queue.len() <= 16,
                    "the string encoding only supports 16 slots of lookback"
                );
                let parent: i32 = item.parent_idx.map(|x| x + 1).unwrap_or(0).try_into().unwrap();
                if let Some(idx) = parents_backref_queue.iter().position(|p: &i32| *p == parent) {
                    parents.push(
                        char::try_from('0' as u32 + u32::try_from(idx).unwrap())
                            .expect("last possible value is '?'"),
                    );
                } else if parent == 0 {
                    write_vlqhex_to_string(parent, &mut parents);
                } else {
                    parents_backref_queue.push_front(parent);
                    write_vlqhex_to_string(parent, &mut parents);
                    if parents_backref_queue.len() > 16 {
                        parents_backref_queue.pop_back();
                    }
                }

                if Some(item.name.as_str()) == last_name {
                    names.push("");
                } else {
                    names.push(item.name.as_str());
                    last_name = Some(item.name.as_str());
                }

                if !item.path.is_empty() {
                    full_paths.push((index, &item.path));
                }

                match &item.search_type {
                    Some(ty) => ty.write_to_string(&mut functions, &mut type_backref_queue),
                    None => functions.push('`'),
                }

                if item.deprecation.is_some() {
                    // bitmasks always use 1-indexing for items, with 0 as the crate itself
                    deprecated.push(u32::try_from(index + 1).unwrap());
                }
            }

            for (index, path) in &revert_extra_paths {
                full_paths.push((*index, path));
            }

            let param_names: Vec<(usize, String)> = {
                let mut prev = Vec::new();
                let mut result = Vec::new();
                for (index, item) in self.items.iter().enumerate() {
                    if let Some(ty) = &item.search_type
                        && let my = ty
                            .param_names
                            .iter()
                            .filter_map(|sym| sym.map(|sym| sym.to_string()))
                            .collect::<Vec<_>>()
                        && my != prev
                    {
                        result.push((index, my.join(",")));
                        prev = my;
                    }
                }
                result
            };

            let has_aliases = !self.aliases.is_empty();
            let mut crate_data =
                serializer.serialize_struct("CrateData", if has_aliases { 13 } else { 12 })?;
            crate_data.serialize_field("t", &types)?;
            crate_data.serialize_field("n", &names)?;
            crate_data.serialize_field("q", &full_paths)?;
            crate_data.serialize_field("i", &parents)?;
            crate_data.serialize_field("f", &functions)?;
            crate_data.serialize_field("D", &self.desc_index)?;
            crate_data.serialize_field("p", &paths)?;
            crate_data.serialize_field("r", &re_exports)?;
            crate_data.serialize_field("b", &self.associated_item_disambiguators)?;
            crate_data.serialize_field("c", &bitmap_to_string(&deprecated))?;
            crate_data.serialize_field("e", &bitmap_to_string(&self.empty_desc))?;
            crate_data.serialize_field("P", &param_names)?;
            if has_aliases {
                crate_data.serialize_field("a", &self.aliases)?;
            }
            crate_data.end()
        }
    }

    let (empty_desc, desc) = {
        let mut empty_desc = Vec::new();
        let mut result = Vec::new();
        let mut set = String::new();
        let mut len: usize = 0;
        let mut item_index: u32 = 0;
        for desc in std::iter::once(&crate_doc).chain(crate_items.iter().map(|item| &item.desc)) {
            if desc.is_empty() {
                empty_desc.push(item_index);
                item_index += 1;
                continue;
            }
            if set.len() >= DESC_INDEX_SHARD_LEN {
                result.push((len, std::mem::take(&mut set)));
                len = 0;
            } else if len != 0 {
                set.push('\n');
            }
            set.push_str(desc);
            len += 1;
            item_index += 1;
        }
        result.push((len, std::mem::take(&mut set)));
        (empty_desc, result)
    };

    let desc_index = {
        let mut desc_index = String::with_capacity(desc.len() * 4);
        for &(len, _) in desc.iter() {
            write_vlqhex_to_string(len.try_into().unwrap(), &mut desc_index);
        }
        desc_index
    };

    assert_eq!(
        crate_items.len() + 1,
        desc.iter().map(|(len, _)| *len).sum::<usize>() + empty_desc.len()
    );

    // The index, which is actually used to search, is JSON
    // It uses `JSON.parse(..)` to actually load, since JSON
    // parses faster than the full JavaScript syntax.
    let crate_name = krate.name(tcx);
    let data = CrateData {
        items: crate_items,
        paths: crate_paths,
        aliases: &aliases,
        associated_item_disambiguators: &associated_item_disambiguators,
        desc_index,
        empty_desc,
    };
    let index = OrderedJson::array_unsorted([
        OrderedJson::serialize(crate_name.as_str()).unwrap(),
        OrderedJson::serialize(data).unwrap(),
    ]);
    SerializedSearchIndex { index, desc }
}

pub(crate) fn get_function_type_for_search(
    item: &clean::Item,
    tcx: TyCtxt<'_>,
    impl_generics: Option<&(clean::Type, clean::Generics)>,
    parent: Option<DefId>,
    cache: &Cache,
) -> Option<IndexItemFunctionType> {
    let mut trait_info = None;
    let impl_or_trait_generics = impl_generics.or_else(|| {
        if let Some(def_id) = parent
            && let Some(trait_) = cache.traits.get(&def_id)
            && let Some((path, _)) =
                cache.paths.get(&def_id).or_else(|| cache.external_paths.get(&def_id))
        {
            let path = clean::Path {
                res: rustc_hir::def::Res::Def(rustc_hir::def::DefKind::Trait, def_id),
                segments: path
                    .iter()
                    .map(|name| clean::PathSegment {
                        name: *name,
                        args: clean::GenericArgs::AngleBracketed {
                            args: ThinVec::new(),
                            constraints: ThinVec::new(),
                        },
                    })
                    .collect(),
            };
            trait_info = Some((clean::Type::Path { path }, trait_.generics.clone()));
            Some(trait_info.as_ref().unwrap())
        } else {
            None
        }
    });
    let (mut inputs, mut output, param_names, where_clause) = match item.kind {
        clean::ForeignFunctionItem(ref f, _)
        | clean::FunctionItem(ref f)
        | clean::MethodItem(ref f, _)
        | clean::RequiredMethodItem(ref f) => {
            get_fn_inputs_and_outputs(f, tcx, impl_or_trait_generics, cache)
        }
        clean::ConstantItem(ref c) => make_nullary_fn(&c.type_),
        clean::StaticItem(ref s) => make_nullary_fn(&s.type_),
        clean::StructFieldItem(ref t) if let Some(parent) = parent => {
            let mut rgen: FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)> =
                Default::default();
            let output = get_index_type(t, vec![], &mut rgen);
            let input = RenderType {
                id: Some(RenderTypeId::DefId(parent)),
                generics: None,
                bindings: None,
            };
            (vec![input], vec![output], vec![], vec![])
        }
        _ => return None,
    };

    inputs.retain(|a| a.id.is_some() || a.generics.is_some());
    output.retain(|a| a.id.is_some() || a.generics.is_some());

    Some(IndexItemFunctionType { inputs, output, where_clause, param_names })
}

fn get_index_type(
    clean_type: &clean::Type,
    generics: Vec<RenderType>,
    rgen: &mut FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)>,
) -> RenderType {
    RenderType {
        id: get_index_type_id(clean_type, rgen),
        generics: if generics.is_empty() { None } else { Some(generics) },
        bindings: None,
    }
}

fn get_index_type_id(
    clean_type: &clean::Type,
    rgen: &mut FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)>,
) -> Option<RenderTypeId> {
    use rustc_hir::def::{DefKind, Res};
    match *clean_type {
        clean::Type::Path { ref path, .. } => Some(RenderTypeId::DefId(path.def_id())),
        clean::DynTrait(ref bounds, _) => {
            bounds.first().map(|b| RenderTypeId::DefId(b.trait_.def_id()))
        }
        clean::Primitive(p) => Some(RenderTypeId::Primitive(p)),
        clean::BorrowedRef { .. } => Some(RenderTypeId::Primitive(clean::PrimitiveType::Reference)),
        clean::RawPointer(_, ref type_) => get_index_type_id(type_, rgen),
        // The type parameters are converted to generics in `simplify_fn_type`
        clean::Slice(_) => Some(RenderTypeId::Primitive(clean::PrimitiveType::Slice)),
        clean::Array(_, _) => Some(RenderTypeId::Primitive(clean::PrimitiveType::Array)),
        clean::BareFunction(_) => Some(RenderTypeId::Primitive(clean::PrimitiveType::Fn)),
        clean::Tuple(ref n) if n.is_empty() => {
            Some(RenderTypeId::Primitive(clean::PrimitiveType::Unit))
        }
        clean::Tuple(_) => Some(RenderTypeId::Primitive(clean::PrimitiveType::Tuple)),
        clean::QPath(ref data) => {
            if data.self_type.is_self_type()
                && let Some(clean::Path { res: Res::Def(DefKind::Trait, trait_), .. }) = data.trait_
            {
                let idx = -isize::try_from(rgen.len() + 1).unwrap();
                let (idx, _) = rgen
                    .entry(SimplifiedParam::AssociatedType(trait_, data.assoc.name))
                    .or_insert_with(|| (idx, Vec::new()));
                Some(RenderTypeId::Index(*idx))
            } else {
                None
            }
        }
        // Not supported yet
        clean::Type::Pat(..)
        | clean::Generic(_)
        | clean::SelfTy
        | clean::ImplTrait(_)
        | clean::Infer
        | clean::UnsafeBinder(_) => None,
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum SimplifiedParam {
    // other kinds of type parameters are identified by their name
    Symbol(Symbol),
    // every argument-position impl trait is its own type parameter
    Anonymous(isize),
    // in a trait definition, the associated types are all bound to
    // their own type parameter
    AssociatedType(DefId, Symbol),
}

/// The point of this function is to lower generics and types into the simplified form that the
/// frontend search engine can use.
///
/// For example, `[T, U, i32]]` where you have the bounds: `T: Display, U: Option<T>` will return
/// `[-1, -2, i32] where -1: Display, -2: Option<-1>`. If a type parameter has no traid bound, it
/// will still get a number. If a constraint is present but not used in the actual types, it will
/// not be added to the map.
///
/// This function also works recursively.
#[instrument(level = "trace", skip(tcx, res, rgen, cache))]
fn simplify_fn_type<'a, 'tcx>(
    self_: Option<&'a Type>,
    generics: &Generics,
    arg: &'a Type,
    tcx: TyCtxt<'tcx>,
    recurse: usize,
    res: &mut Vec<RenderType>,
    rgen: &mut FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)>,
    is_return: bool,
    cache: &Cache,
) {
    if recurse >= 10 {
        // FIXME: remove this whole recurse thing when the recursion bug is fixed
        // See #59502 for the original issue.
        return;
    }

    // First, check if it's "Self".
    let (is_self, arg) = if let Some(self_) = self_
        && arg.is_self_type()
    {
        (true, self_)
    } else {
        (false, arg)
    };

    // If this argument is a type parameter and not a trait bound or a type, we need to look
    // for its bounds.
    match *arg {
        Type::Generic(arg_s) => {
            // First we check if the bounds are in a `where` predicate...
            let mut type_bounds = Vec::new();
            for where_pred in generics.where_predicates.iter().filter(|g| match g {
                WherePredicate::BoundPredicate { ty, .. } => *ty == *arg,
                _ => false,
            }) {
                let bounds = where_pred.get_bounds().unwrap_or(&[]);
                for bound in bounds.iter() {
                    if let Some(path) = bound.get_trait_path() {
                        let ty = Type::Path { path };
                        simplify_fn_type(
                            self_,
                            generics,
                            &ty,
                            tcx,
                            recurse + 1,
                            &mut type_bounds,
                            rgen,
                            is_return,
                            cache,
                        );
                    }
                }
            }
            // Otherwise we check if the trait bounds are "inlined" like `T: Option<u32>`...
            if let Some(bound) = generics.params.iter().find(|g| g.is_type() && g.name == arg_s) {
                for bound in bound.get_bounds().unwrap_or(&[]) {
                    if let Some(path) = bound.get_trait_path() {
                        let ty = Type::Path { path };
                        simplify_fn_type(
                            self_,
                            generics,
                            &ty,
                            tcx,
                            recurse + 1,
                            &mut type_bounds,
                            rgen,
                            is_return,
                            cache,
                        );
                    }
                }
            }
            if let Some((idx, _)) = rgen.get(&SimplifiedParam::Symbol(arg_s)) {
                res.push(RenderType {
                    id: Some(RenderTypeId::Index(*idx)),
                    generics: None,
                    bindings: None,
                });
            } else {
                let idx = -isize::try_from(rgen.len() + 1).unwrap();
                rgen.insert(SimplifiedParam::Symbol(arg_s), (idx, type_bounds));
                res.push(RenderType {
                    id: Some(RenderTypeId::Index(idx)),
                    generics: None,
                    bindings: None,
                });
            }
        }
        Type::ImplTrait(ref bounds) => {
            let mut type_bounds = Vec::new();
            for bound in bounds {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    simplify_fn_type(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut type_bounds,
                        rgen,
                        is_return,
                        cache,
                    );
                }
            }
            if is_return && !type_bounds.is_empty() {
                // In return position, `impl Trait` is a unique thing.
                res.push(RenderType { id: None, generics: Some(type_bounds), bindings: None });
            } else {
                // In parameter position, `impl Trait` is the same as an unnamed generic parameter.
                let idx = -isize::try_from(rgen.len() + 1).unwrap();
                rgen.insert(SimplifiedParam::Anonymous(idx), (idx, type_bounds));
                res.push(RenderType {
                    id: Some(RenderTypeId::Index(idx)),
                    generics: None,
                    bindings: None,
                });
            }
        }
        Type::Slice(ref ty) => {
            let mut ty_generics = Vec::new();
            simplify_fn_type(
                self_,
                generics,
                ty,
                tcx,
                recurse + 1,
                &mut ty_generics,
                rgen,
                is_return,
                cache,
            );
            res.push(get_index_type(arg, ty_generics, rgen));
        }
        Type::Array(ref ty, _) => {
            let mut ty_generics = Vec::new();
            simplify_fn_type(
                self_,
                generics,
                ty,
                tcx,
                recurse + 1,
                &mut ty_generics,
                rgen,
                is_return,
                cache,
            );
            res.push(get_index_type(arg, ty_generics, rgen));
        }
        Type::Tuple(ref tys) => {
            let mut ty_generics = Vec::new();
            for ty in tys {
                simplify_fn_type(
                    self_,
                    generics,
                    ty,
                    tcx,
                    recurse + 1,
                    &mut ty_generics,
                    rgen,
                    is_return,
                    cache,
                );
            }
            res.push(get_index_type(arg, ty_generics, rgen));
        }
        Type::BareFunction(ref bf) => {
            let mut ty_generics = Vec::new();
            for ty in bf.decl.inputs.iter().map(|arg| &arg.type_) {
                simplify_fn_type(
                    self_,
                    generics,
                    ty,
                    tcx,
                    recurse + 1,
                    &mut ty_generics,
                    rgen,
                    is_return,
                    cache,
                );
            }
            // The search index, for simplicity's sake, represents fn pointers and closures
            // the same way: as a tuple for the parameters, and an associated type for the
            // return type.
            let mut ty_output = Vec::new();
            simplify_fn_type(
                self_,
                generics,
                &bf.decl.output,
                tcx,
                recurse + 1,
                &mut ty_output,
                rgen,
                is_return,
                cache,
            );
            let ty_bindings = vec![(RenderTypeId::AssociatedType(sym::Output), ty_output)];
            res.push(RenderType {
                id: get_index_type_id(arg, rgen),
                bindings: Some(ty_bindings),
                generics: Some(ty_generics),
            });
        }
        Type::BorrowedRef { lifetime: _, mutability, ref type_ } => {
            let mut ty_generics = Vec::new();
            if mutability.is_mut() {
                ty_generics.push(RenderType {
                    id: Some(RenderTypeId::Mut),
                    generics: None,
                    bindings: None,
                });
            }
            simplify_fn_type(
                self_,
                generics,
                type_,
                tcx,
                recurse + 1,
                &mut ty_generics,
                rgen,
                is_return,
                cache,
            );
            res.push(get_index_type(arg, ty_generics, rgen));
        }
        _ => {
            // This is not a type parameter. So for example if we have `T, U: Option<T>`, and we're
            // looking at `Option`, we enter this "else" condition, otherwise if it's `T`, we don't.
            //
            // So in here, we can add it directly and look for its own type parameters (so for `Option`,
            // we will look for them but not for `T`).
            let mut ty_generics = Vec::new();
            let mut ty_constraints = Vec::new();
            if let Some(arg_generics) = arg.generic_args() {
                for ty in arg_generics.into_iter().filter_map(|param| match param {
                    clean::GenericArg::Type(ty) => Some(ty),
                    _ => None,
                }) {
                    simplify_fn_type(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_generics,
                        rgen,
                        is_return,
                        cache,
                    );
                }
                for constraint in arg_generics.constraints() {
                    simplify_fn_constraint(
                        self_,
                        generics,
                        &constraint,
                        tcx,
                        recurse + 1,
                        &mut ty_constraints,
                        rgen,
                        is_return,
                        cache,
                    );
                }
            }
            // Every trait associated type on self gets assigned to a type parameter index
            // this same one is used later for any appearances of these types
            //
            // for example, Iterator::next is:
            //
            //     trait Iterator {
            //         fn next(&mut self) -> Option<Self::Item>
            //     }
            //
            // Self is technically just Iterator, but we want to pretend it's more like this:
            //
            //     fn next<T>(self: Iterator<Item=T>) -> Option<T>
            if is_self
                && let Type::Path { path } = arg
                && let def_id = path.def_id()
                && let Some(trait_) = cache.traits.get(&def_id)
                && trait_.items.iter().any(|at| at.is_required_associated_type())
            {
                for assoc_ty in &trait_.items {
                    if let clean::ItemKind::RequiredAssocTypeItem(_generics, bounds) =
                        &assoc_ty.kind
                        && let Some(name) = assoc_ty.name
                    {
                        let idx = -isize::try_from(rgen.len() + 1).unwrap();
                        let (idx, stored_bounds) = rgen
                            .entry(SimplifiedParam::AssociatedType(def_id, name))
                            .or_insert_with(|| (idx, Vec::new()));
                        let idx = *idx;
                        if stored_bounds.is_empty() {
                            // Can't just pass stored_bounds to simplify_fn_type,
                            // because it also accepts rgen as a parameter.
                            // Instead, have it fill in this local, then copy it into the map afterward.
                            let mut type_bounds = Vec::new();
                            for bound in bounds {
                                if let Some(path) = bound.get_trait_path() {
                                    let ty = Type::Path { path };
                                    simplify_fn_type(
                                        self_,
                                        generics,
                                        &ty,
                                        tcx,
                                        recurse + 1,
                                        &mut type_bounds,
                                        rgen,
                                        is_return,
                                        cache,
                                    );
                                }
                            }
                            let stored_bounds = &mut rgen
                                .get_mut(&SimplifiedParam::AssociatedType(def_id, name))
                                .unwrap()
                                .1;
                            if stored_bounds.is_empty() {
                                *stored_bounds = type_bounds;
                            }
                        }
                        ty_constraints.push((
                            RenderTypeId::AssociatedType(name),
                            vec![RenderType {
                                id: Some(RenderTypeId::Index(idx)),
                                generics: None,
                                bindings: None,
                            }],
                        ))
                    }
                }
            }
            let id = get_index_type_id(arg, rgen);
            if id.is_some() || !ty_generics.is_empty() {
                res.push(RenderType {
                    id,
                    bindings: if ty_constraints.is_empty() { None } else { Some(ty_constraints) },
                    generics: if ty_generics.is_empty() { None } else { Some(ty_generics) },
                });
            }
        }
    }
}

fn simplify_fn_constraint<'a>(
    self_: Option<&'a Type>,
    generics: &Generics,
    constraint: &'a clean::AssocItemConstraint,
    tcx: TyCtxt<'_>,
    recurse: usize,
    res: &mut Vec<(RenderTypeId, Vec<RenderType>)>,
    rgen: &mut FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)>,
    is_return: bool,
    cache: &Cache,
) {
    let mut ty_constraints = Vec::new();
    let ty_constrained_assoc = RenderTypeId::AssociatedType(constraint.assoc.name);
    for param in &constraint.assoc.args {
        match param {
            clean::GenericArg::Type(arg) => simplify_fn_type(
                self_,
                generics,
                &arg,
                tcx,
                recurse + 1,
                &mut ty_constraints,
                rgen,
                is_return,
                cache,
            ),
            clean::GenericArg::Lifetime(_)
            | clean::GenericArg::Const(_)
            | clean::GenericArg::Infer => {}
        }
    }
    for constraint in constraint.assoc.args.constraints() {
        simplify_fn_constraint(
            self_,
            generics,
            &constraint,
            tcx,
            recurse + 1,
            res,
            rgen,
            is_return,
            cache,
        );
    }
    match &constraint.kind {
        clean::AssocItemConstraintKind::Equality { term } => {
            if let clean::Term::Type(arg) = &term {
                simplify_fn_type(
                    self_,
                    generics,
                    arg,
                    tcx,
                    recurse + 1,
                    &mut ty_constraints,
                    rgen,
                    is_return,
                    cache,
                );
            }
        }
        clean::AssocItemConstraintKind::Bound { bounds } => {
            for bound in &bounds[..] {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    simplify_fn_type(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_constraints,
                        rgen,
                        is_return,
                        cache,
                    );
                }
            }
        }
    }
    res.push((ty_constrained_assoc, ty_constraints));
}

/// Create a fake nullary function.
///
/// Used to allow type-based search on constants and statics.
fn make_nullary_fn(
    clean_type: &clean::Type,
) -> (Vec<RenderType>, Vec<RenderType>, Vec<Option<Symbol>>, Vec<Vec<RenderType>>) {
    let mut rgen: FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)> = Default::default();
    let output = get_index_type(clean_type, vec![], &mut rgen);
    (vec![], vec![output], vec![], vec![])
}

/// Return the full list of types when bounds have been resolved.
///
/// i.e. `fn foo<A: Display, B: Option<A>>(x: u32, y: B)` will return
/// `[u32, Display, Option]`.
fn get_fn_inputs_and_outputs(
    func: &Function,
    tcx: TyCtxt<'_>,
    impl_or_trait_generics: Option<&(clean::Type, clean::Generics)>,
    cache: &Cache,
) -> (Vec<RenderType>, Vec<RenderType>, Vec<Option<Symbol>>, Vec<Vec<RenderType>>) {
    let decl = &func.decl;

    let mut rgen: FxIndexMap<SimplifiedParam, (isize, Vec<RenderType>)> = Default::default();

    let combined_generics;
    let (self_, generics) = if let Some((impl_self, impl_generics)) = impl_or_trait_generics {
        match (impl_generics.is_empty(), func.generics.is_empty()) {
            (true, _) => (Some(impl_self), &func.generics),
            (_, true) => (Some(impl_self), impl_generics),
            (false, false) => {
                let params =
                    func.generics.params.iter().chain(&impl_generics.params).cloned().collect();
                let where_predicates = func
                    .generics
                    .where_predicates
                    .iter()
                    .chain(&impl_generics.where_predicates)
                    .cloned()
                    .collect();
                combined_generics = clean::Generics { params, where_predicates };
                (Some(impl_self), &combined_generics)
            }
        }
    } else {
        (None, &func.generics)
    };

    let mut param_types = Vec::new();
    for param in decl.inputs.iter() {
        simplify_fn_type(
            self_,
            generics,
            &param.type_,
            tcx,
            0,
            &mut param_types,
            &mut rgen,
            false,
            cache,
        );
    }

    let mut ret_types = Vec::new();
    simplify_fn_type(self_, generics, &decl.output, tcx, 0, &mut ret_types, &mut rgen, true, cache);

    let mut simplified_params = rgen.into_iter().collect::<Vec<_>>();
    simplified_params.sort_by_key(|(_, (idx, _))| -idx);
    (
        param_types,
        ret_types,
        simplified_params
            .iter()
            .map(|(name, (_idx, _traits))| match name {
                SimplifiedParam::Symbol(name) => Some(*name),
                SimplifiedParam::Anonymous(_) => None,
                SimplifiedParam::AssociatedType(def_id, name) => {
                    Some(Symbol::intern(&format!("{}::{}", tcx.item_name(*def_id), name)))
                }
            })
            .collect(),
        simplified_params.into_iter().map(|(_name, (_idx, traits))| traits).collect(),
    )
}
