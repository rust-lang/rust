pub(crate) mod encode;

use std::collections::BTreeSet;
use std::collections::hash_map::Entry;
use std::path::Path;

use rustc_ast::join_path_syms;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::sym;
use rustc_span::symbol::{Symbol, kw};
use serde::de::{self, Deserializer, Error as _};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use stringdex::internals as stringdex_internals;
use thin_vec::ThinVec;
use tracing::instrument;

use crate::clean::types::{Function, Generics, ItemId, Type, WherePredicate};
use crate::clean::{self, utils};
use crate::error::Error;
use crate::formats::cache::{Cache, OrphanImplItem};
use crate::formats::item_type::ItemType;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::{self, IndexItem, IndexItemFunctionType, RenderType, RenderTypeId};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct SerializedSearchIndex {
    // data from disk
    names: Vec<String>,
    path_data: Vec<Option<PathData>>,
    entry_data: Vec<Option<EntryData>>,
    descs: Vec<String>,
    function_data: Vec<Option<FunctionData>>,
    alias_pointers: Vec<Option<usize>>,
    // inverted index for concrete types and generics
    type_data: Vec<Option<TypeData>>,
    /// inverted index of generics
    ///
    /// - The outermost list has one entry per alpha-normalized generic.
    ///
    /// - The second layer is sorted by number of types that appear in the
    ///   type signature. The search engine iterates over these in order from
    ///   smallest to largest. Functions with less stuff in their type
    ///   signature are more likely to be what the user wants, because we never
    ///   show functions that are *missing* parts of the query, so removing..
    ///
    /// - The final layer is the list of functions.
    generic_inverted_index: Vec<Vec<Vec<u32>>>,
    // generated in-memory backref cache
    #[serde(skip)]
    crate_paths_index: FxHashMap<(ItemType, Vec<Symbol>), usize>,
}

impl SerializedSearchIndex {
    fn load(doc_root: &Path, resource_suffix: &str) -> Result<SerializedSearchIndex, Error> {
        let mut names: Vec<String> = Vec::new();
        let mut path_data: Vec<Option<PathData>> = Vec::new();
        let mut entry_data: Vec<Option<EntryData>> = Vec::new();
        let mut descs: Vec<String> = Vec::new();
        let mut function_data: Vec<Option<FunctionData>> = Vec::new();
        let mut type_data: Vec<Option<TypeData>> = Vec::new();
        let mut alias_pointers: Vec<Option<usize>> = Vec::new();

        let mut generic_inverted_index: Vec<Vec<Vec<u32>>> = Vec::new();

        match perform_read_strings(resource_suffix, doc_root, "name", &mut names) {
            Ok(()) => {
                perform_read_serde(resource_suffix, doc_root, "path", &mut path_data)?;
                perform_read_serde(resource_suffix, doc_root, "entry", &mut entry_data)?;
                perform_read_strings(resource_suffix, doc_root, "desc", &mut descs)?;
                perform_read_serde(resource_suffix, doc_root, "function", &mut function_data)?;
                perform_read_serde(resource_suffix, doc_root, "type", &mut type_data)?;
                perform_read_serde(resource_suffix, doc_root, "alias", &mut alias_pointers)?;
                perform_read_postings(
                    resource_suffix,
                    doc_root,
                    "generic_inverted_index",
                    &mut generic_inverted_index,
                )?;
            }
            Err(_) => {
                names.clear();
            }
        }
        fn perform_read_strings(
            resource_suffix: &str,
            doc_root: &Path,
            column_name: &str,
            column: &mut Vec<String>,
        ) -> Result<(), Error> {
            let root_path = doc_root.join(format!("search.index/root{resource_suffix}.js"));
            let column_path = doc_root.join(format!("search.index/{column_name}/"));
            stringdex_internals::read_data_from_disk_column(
                root_path,
                column_name.as_bytes(),
                column_path.clone(),
                &mut |_id, item| {
                    column.push(String::from_utf8(item.to_vec())?);
                    Ok(())
                },
            )
            .map_err(
                |error: stringdex_internals::ReadDataError<Box<dyn std::error::Error>>| Error {
                    file: column_path,
                    error: format!("failed to read column from disk: {error}"),
                },
            )
        }
        fn perform_read_serde(
            resource_suffix: &str,
            doc_root: &Path,
            column_name: &str,
            column: &mut Vec<Option<impl for<'de> Deserialize<'de> + 'static>>,
        ) -> Result<(), Error> {
            let root_path = doc_root.join(format!("search.index/root{resource_suffix}.js"));
            let column_path = doc_root.join(format!("search.index/{column_name}/"));
            stringdex_internals::read_data_from_disk_column(
                root_path,
                column_name.as_bytes(),
                column_path.clone(),
                &mut |_id, item| {
                    if item.is_empty() {
                        column.push(None);
                    } else {
                        column.push(Some(serde_json::from_slice(item)?));
                    }
                    Ok(())
                },
            )
            .map_err(
                |error: stringdex_internals::ReadDataError<Box<dyn std::error::Error>>| Error {
                    file: column_path,
                    error: format!("failed to read column from disk: {error}"),
                },
            )
        }
        fn perform_read_postings(
            resource_suffix: &str,
            doc_root: &Path,
            column_name: &str,
            column: &mut Vec<Vec<Vec<u32>>>,
        ) -> Result<(), Error> {
            let root_path = doc_root.join(format!("search.index/root{resource_suffix}.js"));
            let column_path = doc_root.join(format!("search.index/{column_name}/"));
            stringdex_internals::read_data_from_disk_column(
                root_path,
                column_name.as_bytes(),
                column_path.clone(),
                &mut |_id, buf| {
                    let mut postings = Vec::new();
                    encode::read_postings_from_string(&mut postings, buf);
                    column.push(postings);
                    Ok(())
                },
            )
            .map_err(
                |error: stringdex_internals::ReadDataError<Box<dyn std::error::Error>>| Error {
                    file: column_path,
                    error: format!("failed to read column from disk: {error}"),
                },
            )
        }

        assert_eq!(names.len(), path_data.len());
        assert_eq!(path_data.len(), entry_data.len());
        assert_eq!(entry_data.len(), descs.len());
        assert_eq!(descs.len(), function_data.len());
        assert_eq!(function_data.len(), type_data.len());
        assert_eq!(type_data.len(), alias_pointers.len());

        // generic_inverted_index is not the same length as other columns,
        // because it's actually a completely different set of objects

        let mut crate_paths_index: FxHashMap<(ItemType, Vec<Symbol>), usize> = FxHashMap::default();
        for (i, (name, path_data)) in names.iter().zip(path_data.iter()).enumerate() {
            if let Some(path_data) = path_data {
                let full_path = if path_data.module_path.is_empty() {
                    vec![Symbol::intern(name)]
                } else {
                    let mut full_path = path_data.module_path.to_vec();
                    full_path.push(Symbol::intern(name));
                    full_path
                };
                crate_paths_index.insert((path_data.ty, full_path), i);
            }
        }

        Ok(SerializedSearchIndex {
            names,
            path_data,
            entry_data,
            descs,
            function_data,
            type_data,
            alias_pointers,
            generic_inverted_index,
            crate_paths_index,
        })
    }
    fn push(
        &mut self,
        name: String,
        path_data: Option<PathData>,
        entry_data: Option<EntryData>,
        desc: String,
        function_data: Option<FunctionData>,
        type_data: Option<TypeData>,
        alias_pointer: Option<usize>,
    ) -> usize {
        let index = self.names.len();
        assert_eq!(self.names.len(), self.path_data.len());
        if let Some(path_data) = &path_data
            && let name = Symbol::intern(&name)
            && let fqp = if path_data.module_path.is_empty() {
                vec![name]
            } else {
                let mut v = path_data.module_path.clone();
                v.push(name);
                v
            }
            && let Some(&other_path) = self.crate_paths_index.get(&(path_data.ty, fqp))
            && self.path_data.get(other_path).map_or(false, Option::is_some)
        {
            self.path_data.push(None);
        } else {
            self.path_data.push(path_data);
        }
        self.names.push(name);
        assert_eq!(self.entry_data.len(), self.descs.len());
        self.entry_data.push(entry_data);
        assert_eq!(self.descs.len(), self.function_data.len());
        self.descs.push(desc);
        assert_eq!(self.function_data.len(), self.type_data.len());
        self.function_data.push(function_data);
        assert_eq!(self.type_data.len(), self.alias_pointers.len());
        self.type_data.push(type_data);
        self.alias_pointers.push(alias_pointer);
        index
    }
    fn push_path(&mut self, name: String, path_data: PathData) -> usize {
        self.push(name, Some(path_data), None, String::new(), None, None, None)
    }
    fn push_type(&mut self, name: String, path_data: PathData, type_data: TypeData) -> usize {
        self.push(name, Some(path_data), None, String::new(), None, Some(type_data), None)
    }
    fn push_alias(&mut self, name: String, alias_pointer: usize) -> usize {
        self.push(name, None, None, String::new(), None, None, Some(alias_pointer))
    }

    fn get_id_by_module_path(&mut self, path: &[Symbol]) -> usize {
        let ty = if path.len() == 1 { ItemType::ExternCrate } else { ItemType::Module };
        match self.crate_paths_index.entry((ty, path.to_vec())) {
            Entry::Occupied(index) => *index.get(),
            Entry::Vacant(slot) => {
                slot.insert(self.path_data.len());
                let (name, module_path) = path.split_last().unwrap();
                self.push_path(
                    name.as_str().to_string(),
                    PathData { ty, module_path: module_path.to_vec(), exact_module_path: None },
                )
            }
        }
    }

    pub(crate) fn union(mut self, other: &SerializedSearchIndex) -> SerializedSearchIndex {
        let other_entryid_offset = self.names.len();
        let mut map_other_pathid_to_self_pathid: Vec<usize> = Vec::new();
        let mut skips = FxHashSet::default();
        for (other_pathid, other_path_data) in other.path_data.iter().enumerate() {
            if let Some(other_path_data) = other_path_data {
                let mut fqp = other_path_data.module_path.clone();
                let name = Symbol::intern(&other.names[other_pathid]);
                fqp.push(name);
                let self_pathid = other_entryid_offset + other_pathid;
                let self_pathid = match self.crate_paths_index.entry((other_path_data.ty, fqp)) {
                    Entry::Vacant(slot) => {
                        slot.insert(self_pathid);
                        self_pathid
                    }
                    Entry::Occupied(existing_entryid) => {
                        skips.insert(other_pathid);
                        let self_pathid = *existing_entryid.get();
                        let new_type_data = match (
                            self.type_data[self_pathid].take(),
                            other.type_data[other_pathid].as_ref(),
                        ) {
                            (Some(self_type_data), None) => Some(self_type_data),
                            (None, Some(other_type_data)) => Some(TypeData {
                                search_unbox: other_type_data.search_unbox,
                                inverted_function_inputs_index: other_type_data
                                    .inverted_function_inputs_index
                                    .iter()
                                    .cloned()
                                    .map(|mut list: Vec<u32>| {
                                        for fnid in &mut list {
                                            assert!(
                                                other.function_data
                                                    [usize::try_from(*fnid).unwrap()]
                                                .is_some(),
                                            );
                                            // this is valid because we call `self.push()` once, exactly, for every entry,
                                            // even if we're just pushing a tombstone
                                            *fnid += u32::try_from(other_entryid_offset).unwrap();
                                        }
                                        list
                                    })
                                    .collect(),
                                inverted_function_output_index: other_type_data
                                    .inverted_function_output_index
                                    .iter()
                                    .cloned()
                                    .map(|mut list: Vec<u32>| {
                                        for fnid in &mut list {
                                            assert!(
                                                other.function_data
                                                    [usize::try_from(*fnid).unwrap()]
                                                .is_some(),
                                            );
                                            // this is valid because we call `self.push()` once, exactly, for every entry,
                                            // even if we're just pushing a tombstone
                                            *fnid += u32::try_from(other_entryid_offset).unwrap();
                                        }
                                        list
                                    })
                                    .collect(),
                            }),
                            (Some(mut self_type_data), Some(other_type_data)) => {
                                for (size, other_list) in other_type_data
                                    .inverted_function_inputs_index
                                    .iter()
                                    .enumerate()
                                {
                                    while self_type_data.inverted_function_inputs_index.len()
                                        <= size
                                    {
                                        self_type_data
                                            .inverted_function_inputs_index
                                            .push(Vec::new());
                                    }
                                    self_type_data.inverted_function_inputs_index[size].extend(
                                        other_list.iter().copied().map(|fnid| {
                                            assert!(
                                                other.function_data[usize::try_from(fnid).unwrap()]
                                                    .is_some(),
                                            );
                                            // this is valid because we call `self.push()` once, exactly, for every entry,
                                            // even if we're just pushing a tombstone
                                            fnid + u32::try_from(other_entryid_offset).unwrap()
                                        }),
                                    )
                                }
                                for (size, other_list) in other_type_data
                                    .inverted_function_output_index
                                    .iter()
                                    .enumerate()
                                {
                                    while self_type_data.inverted_function_output_index.len()
                                        <= size
                                    {
                                        self_type_data
                                            .inverted_function_output_index
                                            .push(Vec::new());
                                    }
                                    self_type_data.inverted_function_output_index[size].extend(
                                        other_list.iter().copied().map(|fnid| {
                                            assert!(
                                                other.function_data[usize::try_from(fnid).unwrap()]
                                                    .is_some(),
                                            );
                                            // this is valid because we call `self.push()` once, exactly, for every entry,
                                            // even if we're just pushing a tombstone
                                            fnid + u32::try_from(other_entryid_offset).unwrap()
                                        }),
                                    )
                                }
                                Some(self_type_data)
                            }
                            (None, None) => None,
                        };
                        self.type_data[self_pathid] = new_type_data;
                        self_pathid
                    }
                };
                map_other_pathid_to_self_pathid.push(self_pathid);
            } else {
                // if this gets used, we want it to crash
                // this should be impossible as a valid index, since some of the
                // memory must be used for stuff other than the list
                map_other_pathid_to_self_pathid.push(!0);
            }
        }
        for other_entryid in 0..other.names.len() {
            if skips.contains(&other_entryid) {
                // we push tombstone entries to keep the IDs lined up
                self.push(String::new(), None, None, String::new(), None, None, None);
            } else {
                self.push(
                    other.names[other_entryid].clone(),
                    other.path_data[other_entryid].clone(),
                    other.entry_data[other_entryid].as_ref().map(|other_entry_data| EntryData {
                        parent: other_entry_data
                            .parent
                            .map(|parent| map_other_pathid_to_self_pathid[parent])
                            .clone(),
                        module_path: other_entry_data
                            .module_path
                            .map(|path| map_other_pathid_to_self_pathid[path])
                            .clone(),
                        exact_module_path: other_entry_data
                            .exact_module_path
                            .map(|exact_path| map_other_pathid_to_self_pathid[exact_path])
                            .clone(),
                        krate: map_other_pathid_to_self_pathid[other_entry_data.krate],
                        ..other_entry_data.clone()
                    }),
                    other.descs[other_entryid].clone(),
                    other.function_data[other_entryid].as_ref().map(|function_data| FunctionData {
                        function_signature: {
                            let (mut func, _offset) =
                                IndexItemFunctionType::read_from_string_without_param_names(
                                    function_data.function_signature.as_bytes(),
                                );
                            fn map_fn_sig_item(
                                map_other_pathid_to_self_pathid: &mut Vec<usize>,
                                ty: &mut RenderType,
                            ) {
                                match ty.id {
                                    None => {}
                                    Some(RenderTypeId::Index(generic)) if generic < 0 => {}
                                    Some(RenderTypeId::Index(id)) => {
                                        let id = usize::try_from(id).unwrap();
                                        let id = map_other_pathid_to_self_pathid[id];
                                        assert!(id != !0);
                                        ty.id =
                                            Some(RenderTypeId::Index(isize::try_from(id).unwrap()));
                                    }
                                    _ => unreachable!(),
                                }
                                if let Some(generics) = &mut ty.generics {
                                    for generic in generics {
                                        map_fn_sig_item(map_other_pathid_to_self_pathid, generic);
                                    }
                                }
                                if let Some(bindings) = &mut ty.bindings {
                                    for (param, constraints) in bindings {
                                        *param = match *param {
                                            param @ RenderTypeId::Index(generic) if generic < 0 => {
                                                param
                                            }
                                            RenderTypeId::Index(id) => {
                                                let id = usize::try_from(id).unwrap();
                                                let id = map_other_pathid_to_self_pathid[id];
                                                assert!(id != !0);
                                                RenderTypeId::Index(isize::try_from(id).unwrap())
                                            }
                                            _ => unreachable!(),
                                        };
                                        for constraint in constraints {
                                            map_fn_sig_item(
                                                map_other_pathid_to_self_pathid,
                                                constraint,
                                            );
                                        }
                                    }
                                }
                            }
                            for input in &mut func.inputs {
                                map_fn_sig_item(&mut map_other_pathid_to_self_pathid, input);
                            }
                            for output in &mut func.output {
                                map_fn_sig_item(&mut map_other_pathid_to_self_pathid, output);
                            }
                            for clause in &mut func.where_clause {
                                for entry in clause {
                                    map_fn_sig_item(&mut map_other_pathid_to_self_pathid, entry);
                                }
                            }
                            let mut result =
                                String::with_capacity(function_data.function_signature.len());
                            func.write_to_string_without_param_names(&mut result);
                            result
                        },
                        param_names: function_data.param_names.clone(),
                    }),
                    other.type_data[other_entryid].as_ref().map(|type_data| TypeData {
                        inverted_function_inputs_index: type_data
                            .inverted_function_inputs_index
                            .iter()
                            .cloned()
                            .map(|mut list| {
                                for fnid in &mut list {
                                    assert!(
                                        other.function_data[usize::try_from(*fnid).unwrap()]
                                            .is_some(),
                                    );
                                    // this is valid because we call `self.push()` once, exactly, for every entry,
                                    // even if we're just pushing a tombstone
                                    *fnid += u32::try_from(other_entryid_offset).unwrap();
                                }
                                list
                            })
                            .collect(),
                        inverted_function_output_index: type_data
                            .inverted_function_output_index
                            .iter()
                            .cloned()
                            .map(|mut list| {
                                for fnid in &mut list {
                                    assert!(
                                        other.function_data[usize::try_from(*fnid).unwrap()]
                                            .is_some(),
                                    );
                                    // this is valid because we call `self.push()` once, exactly, for every entry,
                                    // even if we're just pushing a tombstone
                                    *fnid += u32::try_from(other_entryid_offset).unwrap();
                                }
                                list
                            })
                            .collect(),
                        search_unbox: type_data.search_unbox,
                    }),
                    other.alias_pointers[other_entryid]
                        .map(|alias_pointer| alias_pointer + other_entryid_offset),
                );
            }
        }
        for (i, other_generic_inverted_index) in other.generic_inverted_index.iter().enumerate() {
            for (size, other_list) in other_generic_inverted_index.iter().enumerate() {
                let self_generic_inverted_index = match self.generic_inverted_index.get_mut(i) {
                    Some(self_generic_inverted_index) => self_generic_inverted_index,
                    None => {
                        self.generic_inverted_index.push(Vec::new());
                        self.generic_inverted_index.last_mut().unwrap()
                    }
                };
                while self_generic_inverted_index.len() <= size {
                    self_generic_inverted_index.push(Vec::new());
                }
                self_generic_inverted_index[size].extend(
                    other_list
                        .iter()
                        .copied()
                        .map(|fnid| fnid + u32::try_from(other_entryid_offset).unwrap()),
                );
            }
        }
        self
    }

    pub(crate) fn sort(self) -> SerializedSearchIndex {
        let mut idlist: Vec<usize> = (0..self.names.len()).collect();
        // nameless entries are tombstones, and will be removed after sorting
        // sort shorter names first, so that we can present them in order out of search.js
        idlist.sort_by_key(|&id| {
            (
                self.names[id].is_empty(),
                self.names[id].len(),
                &self.names[id],
                self.entry_data[id].as_ref().map_or("", |entry| self.names[entry.krate].as_str()),
                self.path_data[id].as_ref().map_or(&[][..], |entry| &entry.module_path[..]),
            )
        });
        let map = FxHashMap::from_iter(
            idlist.iter().enumerate().map(|(new_id, &old_id)| (old_id, new_id)),
        );
        let mut new = SerializedSearchIndex::default();
        for &id in &idlist {
            if self.names[id].is_empty() {
                break;
            }
            new.push(
                self.names[id].clone(),
                self.path_data[id].clone(),
                self.entry_data[id].as_ref().map(
                    |EntryData {
                         krate,
                         ty,
                         module_path,
                         exact_module_path,
                         parent,
                         deprecated,
                         associated_item_disambiguator,
                     }| EntryData {
                        krate: *map.get(krate).unwrap(),
                        ty: *ty,
                        module_path: module_path.and_then(|path_id| map.get(&path_id).copied()),
                        exact_module_path: exact_module_path
                            .and_then(|path_id| map.get(&path_id).copied()),
                        parent: parent.and_then(|path_id| map.get(&path_id).copied()),
                        deprecated: *deprecated,
                        associated_item_disambiguator: associated_item_disambiguator.clone(),
                    },
                ),
                self.descs[id].clone(),
                self.function_data[id].as_ref().map(
                    |FunctionData { function_signature, param_names }| FunctionData {
                        function_signature: {
                            let (mut func, _offset) =
                                IndexItemFunctionType::read_from_string_without_param_names(
                                    function_signature.as_bytes(),
                                );
                            fn map_fn_sig_item(map: &FxHashMap<usize, usize>, ty: &mut RenderType) {
                                match ty.id {
                                    None => {}
                                    Some(RenderTypeId::Index(generic)) if generic < 0 => {}
                                    Some(RenderTypeId::Index(id)) => {
                                        let id = usize::try_from(id).unwrap();
                                        let id = *map.get(&id).unwrap();
                                        assert!(id != !0);
                                        ty.id =
                                            Some(RenderTypeId::Index(isize::try_from(id).unwrap()));
                                    }
                                    _ => unreachable!(),
                                }
                                if let Some(generics) = &mut ty.generics {
                                    for generic in generics {
                                        map_fn_sig_item(map, generic);
                                    }
                                }
                                if let Some(bindings) = &mut ty.bindings {
                                    for (param, constraints) in bindings {
                                        *param = match *param {
                                            param @ RenderTypeId::Index(generic) if generic < 0 => {
                                                param
                                            }
                                            RenderTypeId::Index(id) => {
                                                let id = usize::try_from(id).unwrap();
                                                let id = *map.get(&id).unwrap();
                                                assert!(id != !0);
                                                RenderTypeId::Index(isize::try_from(id).unwrap())
                                            }
                                            _ => unreachable!(),
                                        };
                                        for constraint in constraints {
                                            map_fn_sig_item(map, constraint);
                                        }
                                    }
                                }
                            }
                            for input in &mut func.inputs {
                                map_fn_sig_item(&map, input);
                            }
                            for output in &mut func.output {
                                map_fn_sig_item(&map, output);
                            }
                            for clause in &mut func.where_clause {
                                for entry in clause {
                                    map_fn_sig_item(&map, entry);
                                }
                            }
                            let mut result = String::with_capacity(function_signature.len());
                            func.write_to_string_without_param_names(&mut result);
                            result
                        },
                        param_names: param_names.clone(),
                    },
                ),
                self.type_data[id].as_ref().map(
                    |TypeData {
                         search_unbox,
                         inverted_function_inputs_index,
                         inverted_function_output_index,
                     }| {
                        let inverted_function_inputs_index: Vec<Vec<u32>> =
                            inverted_function_inputs_index
                                .iter()
                                .cloned()
                                .map(|mut list| {
                                    for id in &mut list {
                                        *id = u32::try_from(
                                            *map.get(&usize::try_from(*id).unwrap()).unwrap(),
                                        )
                                        .unwrap();
                                    }
                                    list.sort();
                                    list
                                })
                                .collect();
                        let inverted_function_output_index: Vec<Vec<u32>> =
                            inverted_function_output_index
                                .iter()
                                .cloned()
                                .map(|mut list| {
                                    for id in &mut list {
                                        *id = u32::try_from(
                                            *map.get(&usize::try_from(*id).unwrap()).unwrap(),
                                        )
                                        .unwrap();
                                    }
                                    list.sort();
                                    list
                                })
                                .collect();
                        TypeData {
                            search_unbox: *search_unbox,
                            inverted_function_inputs_index,
                            inverted_function_output_index,
                        }
                    },
                ),
                self.alias_pointers[id].and_then(|alias| map.get(&alias).copied()),
            );
        }
        new.generic_inverted_index = self
            .generic_inverted_index
            .into_iter()
            .map(|mut postings| {
                for list in postings.iter_mut() {
                    let mut new_list: Vec<u32> = list
                        .iter()
                        .copied()
                        .filter_map(|id| u32::try_from(*map.get(&usize::try_from(id).ok()?)?).ok())
                        .collect();
                    new_list.sort();
                    *list = new_list;
                }
                postings
            })
            .collect();
        new
    }

    pub(crate) fn write_to(self, doc_root: &Path, resource_suffix: &str) -> Result<(), Error> {
        let SerializedSearchIndex {
            names,
            path_data,
            entry_data,
            descs,
            function_data,
            type_data,
            alias_pointers,
            generic_inverted_index,
            crate_paths_index: _,
        } = self;
        let mut serialized_root = Vec::new();
        serialized_root.extend_from_slice(br#"rr_('{"normalizedName":{"I":""#);
        let normalized_names = names
            .iter()
            .map(|name| {
                if name.contains("_") {
                    name.replace("_", "").to_ascii_lowercase()
                } else {
                    name.to_ascii_lowercase()
                }
            })
            .collect::<Vec<String>>();
        let names_search_tree = stringdex_internals::tree::encode_search_tree_ukkonen(
            normalized_names.iter().map(|name| name.as_bytes()),
        );
        let dir_path = doc_root.join(format!("search.index/"));
        let _ = std::fs::remove_dir_all(&dir_path); // if already missing, no problem
        stringdex_internals::write_tree_to_disk(
            &names_search_tree,
            &dir_path,
            &mut serialized_root,
        )
        .map_err(|error| Error {
            file: dir_path,
            error: format!("failed to write name tree to disk: {error}"),
        })?;
        std::mem::drop(names_search_tree);
        serialized_root.extend_from_slice(br#"","#);
        serialized_root.extend_from_slice(&perform_write_strings(
            doc_root,
            "normalizedName",
            normalized_names.into_iter(),
        )?);
        serialized_root.extend_from_slice(br#"},"crateNames":{"#);
        let mut crates: Vec<&[u8]> = entry_data
            .iter()
            .filter_map(|entry_data| Some(names[entry_data.as_ref()?.krate].as_bytes()))
            .collect();
        crates.sort();
        crates.dedup();
        serialized_root.extend_from_slice(&perform_write_strings(
            doc_root,
            "crateNames",
            crates.into_iter(),
        )?);
        serialized_root.extend_from_slice(br#"},"name":{"#);
        serialized_root.extend_from_slice(&perform_write_strings(doc_root, "name", names.iter())?);
        serialized_root.extend_from_slice(br#"},"path":{"#);
        serialized_root.extend_from_slice(&perform_write_serde(doc_root, "path", path_data)?);
        serialized_root.extend_from_slice(br#"},"entry":{"#);
        serialized_root.extend_from_slice(&perform_write_serde(doc_root, "entry", entry_data)?);
        serialized_root.extend_from_slice(br#"},"desc":{"#);
        serialized_root.extend_from_slice(&perform_write_strings(
            doc_root,
            "desc",
            descs.into_iter(),
        )?);
        serialized_root.extend_from_slice(br#"},"function":{"#);
        serialized_root.extend_from_slice(&perform_write_serde(
            doc_root,
            "function",
            function_data,
        )?);
        serialized_root.extend_from_slice(br#"},"type":{"#);
        serialized_root.extend_from_slice(&perform_write_serde(doc_root, "type", type_data)?);
        serialized_root.extend_from_slice(br#"},"alias":{"#);
        serialized_root.extend_from_slice(&perform_write_serde(doc_root, "alias", alias_pointers)?);
        serialized_root.extend_from_slice(br#"},"generic_inverted_index":{"#);
        serialized_root.extend_from_slice(&perform_write_postings(
            doc_root,
            "generic_inverted_index",
            generic_inverted_index,
        )?);
        serialized_root.extend_from_slice(br#"}}')"#);
        fn perform_write_strings(
            doc_root: &Path,
            dirname: &str,
            mut column: impl Iterator<Item = impl AsRef<[u8]> + Clone> + ExactSizeIterator,
        ) -> Result<Vec<u8>, Error> {
            let dir_path = doc_root.join(format!("search.index/{dirname}"));
            stringdex_internals::write_data_to_disk(&mut column, &dir_path).map_err(|error| Error {
                file: dir_path,
                error: format!("failed to write column to disk: {error}"),
            })
        }
        fn perform_write_serde(
            doc_root: &Path,
            dirname: &str,
            column: Vec<Option<impl Serialize>>,
        ) -> Result<Vec<u8>, Error> {
            perform_write_strings(
                doc_root,
                dirname,
                column.into_iter().map(|value| {
                    if let Some(value) = value {
                        serde_json::to_vec(&value).unwrap()
                    } else {
                        Vec::new()
                    }
                }),
            )
        }
        fn perform_write_postings(
            doc_root: &Path,
            dirname: &str,
            column: Vec<Vec<Vec<u32>>>,
        ) -> Result<Vec<u8>, Error> {
            perform_write_strings(
                doc_root,
                dirname,
                column.into_iter().map(|postings| {
                    let mut buf = Vec::new();
                    encode::write_postings_to_string(&postings, &mut buf);
                    buf
                }),
            )
        }
        std::fs::write(
            doc_root.join(format!("search.index/root{resource_suffix}.js")),
            serialized_root,
        )
        .map_err(|error| Error {
            file: doc_root.join(format!("search.index/root{resource_suffix}.js")),
            error: format!("failed to write root to disk: {error}"),
        })?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct EntryData {
    krate: usize,
    ty: ItemType,
    module_path: Option<usize>,
    exact_module_path: Option<usize>,
    parent: Option<usize>,
    deprecated: bool,
    associated_item_disambiguator: Option<String>,
}

impl Serialize for EntryData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.krate)?;
        seq.serialize_element(&self.ty)?;
        seq.serialize_element(&self.module_path.map(|id| id + 1).unwrap_or(0))?;
        seq.serialize_element(&self.exact_module_path.map(|id| id + 1).unwrap_or(0))?;
        seq.serialize_element(&self.parent.map(|id| id + 1).unwrap_or(0))?;
        seq.serialize_element(&if self.deprecated { 1 } else { 0 })?;
        if let Some(disambig) = &self.associated_item_disambiguator {
            seq.serialize_element(&disambig)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for EntryData {
    fn deserialize<D>(deserializer: D) -> Result<EntryData, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EntryDataVisitor;
        impl<'de> de::Visitor<'de> for EntryDataVisitor {
            type Value = EntryData;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "path data")
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut v: A) -> Result<EntryData, A::Error> {
                let krate: usize =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("krate"))?;
                let ty: ItemType =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("ty"))?;
                let module_path: SerializedOptional32 =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("module_path"))?;
                let exact_module_path: SerializedOptional32 = v
                    .next_element()?
                    .ok_or_else(|| A::Error::missing_field("exact_module_path"))?;
                let parent: SerializedOptional32 =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("parent"))?;
                let deprecated: u32 = v.next_element()?.unwrap_or(0);
                let associated_item_disambiguator: Option<String> = v.next_element()?;
                Ok(EntryData {
                    krate,
                    ty,
                    module_path: Option::<i32>::from(module_path).map(|path| path as usize),
                    exact_module_path: Option::<i32>::from(exact_module_path)
                        .map(|path| path as usize),
                    parent: Option::<i32>::from(parent).map(|path| path as usize),
                    deprecated: deprecated != 0,
                    associated_item_disambiguator,
                })
            }
        }
        deserializer.deserialize_any(EntryDataVisitor)
    }
}

#[derive(Clone, Debug)]
struct PathData {
    ty: ItemType,
    module_path: Vec<Symbol>,
    exact_module_path: Option<Vec<Symbol>>,
}

impl Serialize for PathData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.ty)?;
        seq.serialize_element(&if self.module_path.is_empty() {
            String::new()
        } else {
            join_path_syms(&self.module_path)
        })?;
        if let Some(ref path) = self.exact_module_path {
            seq.serialize_element(&if path.is_empty() {
                String::new()
            } else {
                join_path_syms(path)
            })?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for PathData {
    fn deserialize<D>(deserializer: D) -> Result<PathData, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PathDataVisitor;
        impl<'de> de::Visitor<'de> for PathDataVisitor {
            type Value = PathData;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "path data")
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut v: A) -> Result<PathData, A::Error> {
                let ty: ItemType =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("ty"))?;
                let module_path: String =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("module_path"))?;
                let exact_module_path: Option<String> =
                    v.next_element()?.and_then(SerializedOptionalString::into);
                Ok(PathData {
                    ty,
                    module_path: if module_path.is_empty() {
                        vec![]
                    } else {
                        module_path.split("::").map(Symbol::intern).collect()
                    },
                    exact_module_path: exact_module_path.map(|path| {
                        if path.is_empty() {
                            vec![]
                        } else {
                            path.split("::").map(Symbol::intern).collect()
                        }
                    }),
                })
            }
        }
        deserializer.deserialize_any(PathDataVisitor)
    }
}

#[derive(Clone, Debug)]
struct TypeData {
    /// If set to "true", the generics can be matched without having to
    /// mention the type itself. The truth table, assuming `Unboxable`
    /// has `search_unbox = true` and `Inner` has `search_unbox = false`
    ///
    /// | **query**          | `Unboxable<Inner>` | `Inner` | `Inner<Unboxable>` |
    /// |--------------------|--------------------|---------|--------------------|
    /// | `Inner`            | yes                | yes     | yes                |
    /// | `Unboxable`        | yes                | no      | no                 |
    /// | `Unboxable<Inner>` | yes                | no      | no                 |
    /// | `Inner<Unboxable>` | no                 | no      | yes                |
    search_unbox: bool,
    /// List of functions that mention this type in their type signature,
    /// on the left side of the `->` arrow.
    ///
    /// - The outer layer is sorted by number of types that appear in the
    ///   type signature. The search engine iterates over these in order from
    ///   smallest to largest. Functions with less stuff in their type
    ///   signature are more likely to be what the user wants, because we never
    ///   show functions that are *missing* parts of the query, so removing..
    ///
    /// - The inner layer is the list of functions.
    inverted_function_inputs_index: Vec<Vec<u32>>,
    /// List of functions that mention this type in their type signature,
    /// on the right side of the `->` arrow.
    inverted_function_output_index: Vec<Vec<u32>>,
}

impl Serialize for TypeData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.search_unbox
            || !self.inverted_function_inputs_index.is_empty()
            || !self.inverted_function_output_index.is_empty()
        {
            let mut seq = serializer.serialize_seq(None)?;
            let mut buf = Vec::new();
            encode::write_postings_to_string(&self.inverted_function_inputs_index, &mut buf);
            let mut serialized_result = Vec::new();
            stringdex_internals::encode::write_base64_to_bytes(&buf, &mut serialized_result);
            seq.serialize_element(&str::from_utf8(&serialized_result).unwrap())?;
            buf.clear();
            serialized_result.clear();
            encode::write_postings_to_string(&self.inverted_function_output_index, &mut buf);
            stringdex_internals::encode::write_base64_to_bytes(&buf, &mut serialized_result);
            seq.serialize_element(&str::from_utf8(&serialized_result).unwrap())?;
            if self.search_unbox {
                seq.serialize_element(&1)?;
            }
            seq.end()
        } else {
            None::<()>.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for TypeData {
    fn deserialize<D>(deserializer: D) -> Result<TypeData, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TypeDataVisitor;
        impl<'de> de::Visitor<'de> for TypeDataVisitor {
            type Value = TypeData;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "type data")
            }
            fn visit_none<E>(self) -> Result<TypeData, E> {
                Ok(TypeData {
                    inverted_function_inputs_index: vec![],
                    inverted_function_output_index: vec![],
                    search_unbox: false,
                })
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut v: A) -> Result<TypeData, A::Error> {
                let inverted_function_inputs_index: String =
                    v.next_element()?.unwrap_or(String::new());
                let inverted_function_output_index: String =
                    v.next_element()?.unwrap_or(String::new());
                let search_unbox: u32 = v.next_element()?.unwrap_or(0);
                let mut idx: Vec<u8> = Vec::new();
                stringdex_internals::decode::read_base64_from_bytes(
                    inverted_function_inputs_index.as_bytes(),
                    &mut idx,
                )
                .unwrap();
                let mut inverted_function_inputs_index = Vec::new();
                encode::read_postings_from_string(&mut inverted_function_inputs_index, &idx);
                idx.clear();
                stringdex_internals::decode::read_base64_from_bytes(
                    inverted_function_output_index.as_bytes(),
                    &mut idx,
                )
                .unwrap();
                let mut inverted_function_output_index = Vec::new();
                encode::read_postings_from_string(&mut inverted_function_output_index, &idx);
                Ok(TypeData {
                    inverted_function_inputs_index,
                    inverted_function_output_index,
                    search_unbox: search_unbox == 1,
                })
            }
        }
        deserializer.deserialize_any(TypeDataVisitor)
    }
}

enum SerializedOptionalString {
    None,
    Some(String),
}

impl From<SerializedOptionalString> for Option<String> {
    fn from(me: SerializedOptionalString) -> Option<String> {
        match me {
            SerializedOptionalString::Some(string) => Some(string),
            SerializedOptionalString::None => None,
        }
    }
}

impl Serialize for SerializedOptionalString {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            SerializedOptionalString::Some(string) => string.serialize(serializer),
            SerializedOptionalString::None => 0.serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for SerializedOptionalString {
    fn deserialize<D>(deserializer: D) -> Result<SerializedOptionalString, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SerializedOptionalStringVisitor;
        impl<'de> de::Visitor<'de> for SerializedOptionalStringVisitor {
            type Value = SerializedOptionalString;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "0 or string")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<SerializedOptionalString, E> {
                if v != 0 {
                    return Err(E::missing_field("not 0"));
                }
                Ok(SerializedOptionalString::None)
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<SerializedOptionalString, E> {
                Ok(SerializedOptionalString::Some(v))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<SerializedOptionalString, E> {
                Ok(SerializedOptionalString::Some(v.to_string()))
            }
        }
        deserializer.deserialize_any(SerializedOptionalStringVisitor)
    }
}

enum SerializedOptional32 {
    None,
    Some(i32),
}

impl From<SerializedOptional32> for Option<i32> {
    fn from(me: SerializedOptional32) -> Option<i32> {
        match me {
            SerializedOptional32::Some(number) => Some(number),
            SerializedOptional32::None => None,
        }
    }
}

impl Serialize for SerializedOptional32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            &SerializedOptional32::Some(number) if number < 0 => number.serialize(serializer),
            &SerializedOptional32::Some(number) => (number + 1).serialize(serializer),
            &SerializedOptional32::None => 0.serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for SerializedOptional32 {
    fn deserialize<D>(deserializer: D) -> Result<SerializedOptional32, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SerializedOptional32Visitor;
        impl<'de> de::Visitor<'de> for SerializedOptional32Visitor {
            type Value = SerializedOptional32;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "integer")
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<SerializedOptional32, E> {
                Ok(match v {
                    0 => SerializedOptional32::None,
                    v if v < 0 => SerializedOptional32::Some(v as i32),
                    v => SerializedOptional32::Some(v as i32 - 1),
                })
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<SerializedOptional32, E> {
                Ok(match v {
                    0 => SerializedOptional32::None,
                    v => SerializedOptional32::Some(v as i32 - 1),
                })
            }
        }
        deserializer.deserialize_any(SerializedOptional32Visitor)
    }
}

#[derive(Clone, Debug)]
pub struct FunctionData {
    function_signature: String,
    param_names: Vec<String>,
}

impl Serialize for FunctionData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(&self.function_signature)?;
        seq.serialize_element(&self.param_names)?;
        seq.end()
    }
}

impl<'de> Deserialize<'de> for FunctionData {
    fn deserialize<D>(deserializer: D) -> Result<FunctionData, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FunctionDataVisitor;
        impl<'de> de::Visitor<'de> for FunctionDataVisitor {
            type Value = FunctionData;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "fn data")
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut v: A) -> Result<FunctionData, A::Error> {
                let function_signature: String = v
                    .next_element()?
                    .ok_or_else(|| A::Error::missing_field("function_signature"))?;
                let param_names: Vec<String> =
                    v.next_element()?.ok_or_else(|| A::Error::missing_field("param_names"))?;
                Ok(FunctionData { function_signature, param_names })
            }
        }
        deserializer.deserialize_any(FunctionDataVisitor)
    }
}

/// Builds the search index from the collected metadata
pub(crate) fn build_index(
    krate: &clean::Crate,
    cache: &mut Cache,
    tcx: TyCtxt<'_>,
    doc_root: &Path,
    resource_suffix: &str,
) -> Result<SerializedSearchIndex, Error> {
    let mut search_index = std::mem::take(&mut cache.search_index);

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &OrphanImplItem { impl_id, parent, ref item, ref impl_generics } in &cache.orphan_impl_items
    {
        if let Some((fqp, _)) = cache.paths.get(&parent) {
            let desc = short_markdown_summary(&item.doc_value(), &item.link_names(cache));
            search_index.push(IndexItem {
                ty: item.type_(),
                defid: item.item_id.as_def_id(),
                name: item.name.unwrap(),
                module_path: fqp[..fqp.len() - 1].to_vec(),
                desc,
                parent: Some(parent),
                parent_idx: None,
                exact_module_path: None,
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

    // Sort search index items. This improves the compressibility of the search index.
    search_index.sort_unstable_by(|k1, k2| {
        // `sort_unstable_by_key` produces lifetime errors
        // HACK(rustdoc): should not be sorting `CrateNum` or `DefIndex`, this will soon go away, too
        let k1 =
            (&k1.module_path, k1.name.as_str(), &k1.ty, k1.parent.map(|id| (id.index, id.krate)));
        let k2 =
            (&k2.module_path, k2.name.as_str(), &k2.ty, k2.parent.map(|id| (id.index, id.krate)));
        Ord::cmp(&k1, &k2)
    });

    // Now, convert to an on-disk search index format
    //
    // if there's already a search index, load it into memory and add the new entries to it
    // otherwise, do nothing
    let mut serialized_index = SerializedSearchIndex::load(doc_root, resource_suffix)?;

    // The crate always goes first in this list
    let crate_name = krate.name(tcx);
    let crate_doc =
        short_markdown_summary(&krate.module.doc_value(), &krate.module.link_names(cache));
    let crate_idx = {
        let crate_path = (ItemType::ExternCrate, vec![crate_name]);
        match serialized_index.crate_paths_index.entry(crate_path) {
            Entry::Occupied(index) => {
                let index = *index.get();
                serialized_index.descs[index] = crate_doc;
                for type_data in serialized_index.type_data.iter_mut() {
                    if let Some(TypeData {
                        inverted_function_inputs_index,
                        inverted_function_output_index,
                        ..
                    }) = type_data
                    {
                        for list in inverted_function_inputs_index
                            .iter_mut()
                            .chain(inverted_function_output_index.iter_mut())
                        {
                            list.retain(|fnid| {
                                serialized_index.entry_data[usize::try_from(*fnid).unwrap()]
                                    .as_ref()
                                    .unwrap()
                                    .krate
                                    != index
                            });
                        }
                    }
                }
                for i in (index + 1)..serialized_index.entry_data.len() {
                    // if this crate has been built before, replace its stuff with new
                    if let Some(EntryData { krate, .. }) = serialized_index.entry_data[i]
                        && krate == index
                    {
                        serialized_index.entry_data[i] = None;
                        serialized_index.descs[i] = String::new();
                        serialized_index.function_data[i] = None;
                        if serialized_index.path_data[i].is_none() {
                            serialized_index.names[i] = String::new();
                        }
                    }
                    if let Some(alias_pointer) = serialized_index.alias_pointers[i]
                        && serialized_index.entry_data[alias_pointer].is_none()
                    {
                        serialized_index.alias_pointers[i] = None;
                        if serialized_index.path_data[i].is_none()
                            && serialized_index.entry_data[i].is_none()
                        {
                            serialized_index.names[i] = String::new();
                        }
                    }
                }
                index
            }
            Entry::Vacant(slot) => {
                let krate = serialized_index.names.len();
                slot.insert(krate);
                serialized_index.push(
                    crate_name.as_str().to_string(),
                    Some(PathData {
                        ty: ItemType::ExternCrate,
                        module_path: vec![],
                        exact_module_path: None,
                    }),
                    Some(EntryData {
                        krate,
                        ty: ItemType::ExternCrate,
                        module_path: None,
                        exact_module_path: None,
                        parent: None,
                        deprecated: false,
                        associated_item_disambiguator: None,
                    }),
                    crate_doc,
                    None,
                    None,
                    None,
                );
                krate
            }
        }
    };

    // First, populate associated item parents
    let crate_items: Vec<&mut IndexItem> = search_index
        .iter_mut()
        .map(|item| {
            item.parent_idx = item.parent.and_then(|defid| {
                cache.paths.get(&defid).map(|&(ref fqp, ty)| {
                    let pathid = serialized_index.names.len();
                    match serialized_index.crate_paths_index.entry((ty, fqp.clone())) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            entry.insert(pathid);
                            let (name, path) = fqp.split_last().unwrap();
                            serialized_index.push_path(
                                name.as_str().to_string(),
                                PathData {
                                    ty,
                                    module_path: path.to_vec(),
                                    exact_module_path: if let Some(exact_path) =
                                        cache.exact_paths.get(&defid)
                                        && let Some((name2, exact_path)) = exact_path.split_last()
                                        && name == name2
                                    {
                                        Some(exact_path.to_vec())
                                    } else {
                                        None
                                    },
                                },
                            );
                            usize::try_from(pathid).unwrap()
                        }
                    }
                })
            });

            if let Some(defid) = item.defid
                && item.parent_idx.is_none()
            {
                // If this is a re-export, retain the original path.
                // Associated items don't use this.
                // Their parent carries the exact fqp instead.
                let exact_fqp = cache
                    .exact_paths
                    .get(&defid)
                    .or_else(|| cache.external_paths.get(&defid).map(|(fqp, _)| fqp));
                item.exact_module_path = exact_fqp.and_then(|fqp| {
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
                            vec![tcx.crate_name(defid.krate)]
                        } else {
                            if fqp.len() < 2 {
                                return None;
                            }
                            fqp[..fqp.len() - 1].to_vec()
                        };
                    if path == item.module_path {
                        return None;
                    }
                    Some(path)
                });
            } else if let Some(parent_idx) = item.parent_idx {
                let i = usize::try_from(parent_idx).unwrap();
                item.module_path =
                    serialized_index.path_data[i].as_ref().unwrap().module_path.clone();
                item.exact_module_path =
                    serialized_index.path_data[i].as_ref().unwrap().exact_module_path.clone();
            }

            &mut *item
        })
        .collect();

    // Now, find anywhere that the same name is used for two different items
    // these need a disambiguator hash for lints
    let mut associated_item_duplicates = FxHashMap::<(usize, ItemType, Symbol), usize>::default();
    for item in crate_items.iter().map(|x| &*x) {
        if item.impl_id.is_some()
            && let Some(parent_idx) = item.parent_idx
        {
            let count =
                associated_item_duplicates.entry((parent_idx, item.ty, item.name)).or_insert(0);
            *count += 1;
        }
    }

    // now populate the actual entries, type data, and function data
    for item in crate_items {
        assert_eq!(
            item.parent.is_some(),
            item.parent_idx.is_some(),
            "`{}` is missing idx",
            item.name
        );

        let module_path = Some(serialized_index.get_id_by_module_path(&item.module_path));
        let exact_module_path = item
            .exact_module_path
            .as_ref()
            .map(|path| serialized_index.get_id_by_module_path(path));

        let new_entry_id = serialized_index.push(
            item.name.as_str().to_string(),
            None,
            Some(EntryData {
                ty: item.ty,
                parent: item.parent_idx,
                module_path,
                exact_module_path,
                deprecated: item.deprecation.is_some(),
                associated_item_disambiguator: if let Some(impl_id) = item.impl_id
                    && let Some(parent_idx) = item.parent_idx
                    && associated_item_duplicates
                        .get(&(parent_idx, item.ty, item.name))
                        .copied()
                        .unwrap_or(0)
                        > 1
                {
                    Some(render::get_id_for_impl(tcx, ItemId::DefId(impl_id)))
                } else {
                    None
                },
                krate: crate_idx,
            }),
            item.desc.to_string(),
            None, // filled in after all the types have been indexed
            None,
            None,
        );

        // Aliases
        // -------
        for alias in &item.aliases[..] {
            serialized_index.push_alias(alias.as_str().to_string(), new_entry_id);
        }

        // Function signature reverse index
        // --------------------------------
        fn insert_into_map(
            ty: ItemType,
            path: &[Symbol],
            exact_path: Option<&[Symbol]>,
            search_unbox: bool,
            serialized_index: &mut SerializedSearchIndex,
            used_in_function_signature: &mut BTreeSet<isize>,
        ) -> RenderTypeId {
            let pathid = serialized_index.names.len();
            let pathid = match serialized_index.crate_paths_index.entry((ty, path.to_vec())) {
                Entry::Occupied(entry) => {
                    let id = *entry.get();
                    if serialized_index.type_data[id].as_mut().is_none() {
                        serialized_index.type_data[id] = Some(TypeData {
                            search_unbox,
                            inverted_function_inputs_index: Vec::new(),
                            inverted_function_output_index: Vec::new(),
                        });
                    } else if search_unbox {
                        serialized_index.type_data[id].as_mut().unwrap().search_unbox = true;
                    }
                    id
                }
                Entry::Vacant(entry) => {
                    entry.insert(pathid);
                    let (name, path) = path.split_last().unwrap();
                    serialized_index.push_type(
                        name.to_string(),
                        PathData {
                            ty,
                            module_path: path.to_vec(),
                            exact_module_path: if let Some(exact_path) = exact_path
                                && let Some((name2, exact_path)) = exact_path.split_last()
                                && name == name2
                            {
                                Some(exact_path.to_vec())
                            } else {
                                None
                            },
                        },
                        TypeData {
                            inverted_function_inputs_index: Vec::new(),
                            inverted_function_output_index: Vec::new(),
                            search_unbox,
                        },
                    );
                    pathid
                }
            };
            used_in_function_signature.insert(isize::try_from(pathid).unwrap());
            RenderTypeId::Index(isize::try_from(pathid).unwrap())
        }

        fn convert_render_type_id(
            id: RenderTypeId,
            cache: &mut Cache,
            serialized_index: &mut SerializedSearchIndex,
            used_in_function_signature: &mut BTreeSet<isize>,
            tcx: TyCtxt<'_>,
        ) -> Option<RenderTypeId> {
            use crate::clean::PrimitiveType;
            let Cache { ref paths, ref external_paths, ref exact_paths, .. } = *cache;
            let search_unbox = match id {
                RenderTypeId::Mut => false,
                RenderTypeId::DefId(defid) => utils::has_doc_flag(tcx, defid, sym::search_unbox),
                RenderTypeId::Primitive(
                    PrimitiveType::Reference | PrimitiveType::RawPointer | PrimitiveType::Tuple,
                ) => true,
                RenderTypeId::Primitive(..) => false,
                RenderTypeId::AssociatedType(..) => false,
                // this bool is only used by `insert_into_map`, so it doesn't matter what we set here
                // because Index means we've already inserted into the map
                RenderTypeId::Index(_) => false,
            };
            match id {
                RenderTypeId::Mut => Some(insert_into_map(
                    ItemType::Keyword,
                    &[kw::Mut],
                    None,
                    search_unbox,
                    serialized_index,
                    used_in_function_signature,
                )),
                RenderTypeId::DefId(defid) => {
                    if let Some(&(ref fqp, item_type)) =
                        paths.get(&defid).or_else(|| external_paths.get(&defid))
                    {
                        if tcx.lang_items().fn_mut_trait() == Some(defid)
                            || tcx.lang_items().fn_once_trait() == Some(defid)
                            || tcx.lang_items().fn_trait() == Some(defid)
                        {
                            let name = *fqp.last().unwrap();
                            // Make absolutely sure we use this single, correct path,
                            // because search.js needs to match. If we don't do this,
                            // there are three different paths that these traits may
                            // appear to come from.
                            Some(insert_into_map(
                                item_type,
                                &[sym::core, sym::ops, name],
                                Some(&[sym::core, sym::ops, name]),
                                search_unbox,
                                serialized_index,
                                used_in_function_signature,
                            ))
                        } else {
                            let exact_fqp = exact_paths
                                .get(&defid)
                                .or_else(|| external_paths.get(&defid).map(|(fqp, _)| fqp))
                                .map(|v| &v[..])
                                // Re-exports only count if the name is exactly the same.
                                // This is a size optimization, since it means we only need
                                // to store the name once (and the path is re-used for everything
                                // exported from this same module). It's also likely to Do
                                // What I Mean, since if a re-export changes the name, it might
                                // also be a change in semantic meaning.
                                .filter(|this_fqp| this_fqp.last() == fqp.last());
                            Some(insert_into_map(
                                item_type,
                                fqp,
                                exact_fqp,
                                search_unbox,
                                serialized_index,
                                used_in_function_signature,
                            ))
                        }
                    } else {
                        None
                    }
                }
                RenderTypeId::Primitive(primitive) => {
                    let sym = primitive.as_sym();
                    Some(insert_into_map(
                        ItemType::Primitive,
                        &[sym],
                        None,
                        search_unbox,
                        serialized_index,
                        used_in_function_signature,
                    ))
                }
                RenderTypeId::Index(index) => {
                    used_in_function_signature.insert(index);
                    Some(id)
                }
                RenderTypeId::AssociatedType(sym) => Some(insert_into_map(
                    ItemType::AssocType,
                    &[sym],
                    None,
                    search_unbox,
                    serialized_index,
                    used_in_function_signature,
                )),
            }
        }

        fn convert_render_type(
            ty: &mut RenderType,
            cache: &mut Cache,
            serialized_index: &mut SerializedSearchIndex,
            used_in_function_signature: &mut BTreeSet<isize>,
            tcx: TyCtxt<'_>,
        ) {
            if let Some(generics) = &mut ty.generics {
                for item in generics {
                    convert_render_type(
                        item,
                        cache,
                        serialized_index,
                        used_in_function_signature,
                        tcx,
                    );
                }
            }
            if let Some(bindings) = &mut ty.bindings {
                bindings.retain_mut(|(associated_type, constraints)| {
                    let converted_associated_type = convert_render_type_id(
                        *associated_type,
                        cache,
                        serialized_index,
                        used_in_function_signature,
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
                            serialized_index,
                            used_in_function_signature,
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
                serialized_index,
                used_in_function_signature,
                tcx,
            );
            use crate::clean::PrimitiveType;
            // These cases are added to the inverted index, but not actually included
            // in the signature. There's a matching set of cases in the
            // `unifyFunctionTypeIsMatchCandidate` function, for the slow path.
            match id {
                // typeNameIdOfArrayOrSlice
                RenderTypeId::Primitive(PrimitiveType::Array | PrimitiveType::Slice) => {
                    insert_into_map(
                        ItemType::Primitive,
                        &[Symbol::intern("[]")],
                        None,
                        false,
                        serialized_index,
                        used_in_function_signature,
                    );
                }
                RenderTypeId::Primitive(PrimitiveType::Tuple | PrimitiveType::Unit) => {
                    // typeNameIdOfArrayOrSlice
                    insert_into_map(
                        ItemType::Primitive,
                        &[Symbol::intern("()")],
                        None,
                        false,
                        serialized_index,
                        used_in_function_signature,
                    );
                }
                // typeNameIdOfHof
                RenderTypeId::Primitive(PrimitiveType::Fn) => {
                    insert_into_map(
                        ItemType::Primitive,
                        &[Symbol::intern("->")],
                        None,
                        false,
                        serialized_index,
                        used_in_function_signature,
                    );
                }
                RenderTypeId::DefId(did)
                    if tcx.lang_items().fn_mut_trait() == Some(did)
                        || tcx.lang_items().fn_once_trait() == Some(did)
                        || tcx.lang_items().fn_trait() == Some(did) =>
                {
                    insert_into_map(
                        ItemType::Primitive,
                        &[Symbol::intern("->")],
                        None,
                        false,
                        serialized_index,
                        used_in_function_signature,
                    );
                }
                // not special
                _ => {}
            }
        }
        if let Some(search_type) = &mut item.search_type {
            let mut used_in_function_inputs = BTreeSet::new();
            let mut used_in_function_output = BTreeSet::new();
            for item in &mut search_type.inputs {
                convert_render_type(
                    item,
                    cache,
                    &mut serialized_index,
                    &mut used_in_function_inputs,
                    tcx,
                );
            }
            for item in &mut search_type.output {
                convert_render_type(
                    item,
                    cache,
                    &mut serialized_index,
                    &mut used_in_function_output,
                    tcx,
                );
            }
            let mut used_in_constraints = Vec::new();
            for constraint in &mut search_type.where_clause {
                let mut used_in_constraint = BTreeSet::new();
                for trait_ in &mut constraint[..] {
                    convert_render_type(
                        trait_,
                        cache,
                        &mut serialized_index,
                        &mut used_in_constraint,
                        tcx,
                    );
                }
                used_in_constraints.push(used_in_constraint);
            }
            loop {
                let mut inserted_any = false;
                for (i, used_in_constraint) in used_in_constraints.iter().enumerate() {
                    let id = !(i as isize);
                    if used_in_function_inputs.contains(&id)
                        && !used_in_function_inputs.is_superset(&used_in_constraint)
                    {
                        used_in_function_inputs.extend(used_in_constraint.iter().copied());
                        inserted_any = true;
                    }
                    if used_in_function_output.contains(&id)
                        && !used_in_function_output.is_superset(&used_in_constraint)
                    {
                        used_in_function_output.extend(used_in_constraint.iter().copied());
                        inserted_any = true;
                    }
                }
                if !inserted_any {
                    break;
                }
            }
            let search_type_size = search_type.size() +
                // Artificially give struct fields a size of 8 instead of their real
                // size of 2. This is because search.js sorts them to the end, so
                // by pushing them down, we prevent them from blocking real 2-arity functions.
                //
                // The number 8 is arbitrary. We want it big, but not enormous,
                // because the postings list has to fill in an empty array for each
                // unoccupied size.
                if item.ty.is_fn_like() { 0 } else { 16 };
            serialized_index.function_data[new_entry_id] = Some(FunctionData {
                function_signature: {
                    let mut function_signature = String::new();
                    search_type.write_to_string_without_param_names(&mut function_signature);
                    function_signature
                },
                param_names: search_type
                    .param_names
                    .iter()
                    .map(|sym| sym.map(|sym| sym.to_string()).unwrap_or(String::new()))
                    .collect::<Vec<String>>(),
            });
            for index in used_in_function_inputs {
                let postings = if index >= 0 {
                    assert!(serialized_index.path_data[index as usize].is_some());
                    &mut serialized_index.type_data[index as usize]
                        .as_mut()
                        .unwrap()
                        .inverted_function_inputs_index
                } else {
                    let generic_id = usize::try_from(-index).unwrap() - 1;
                    for _ in serialized_index.generic_inverted_index.len()..=generic_id {
                        serialized_index.generic_inverted_index.push(Vec::new());
                    }
                    &mut serialized_index.generic_inverted_index[generic_id]
                };
                while postings.len() <= search_type_size {
                    postings.push(Vec::new());
                }
                if postings[search_type_size].last() != Some(&(new_entry_id as u32)) {
                    postings[search_type_size].push(new_entry_id as u32);
                }
            }
            for index in used_in_function_output {
                let postings = if index >= 0 {
                    assert!(serialized_index.path_data[index as usize].is_some());
                    &mut serialized_index.type_data[index as usize]
                        .as_mut()
                        .unwrap()
                        .inverted_function_output_index
                } else {
                    let generic_id = usize::try_from(-index).unwrap() - 1;
                    for _ in serialized_index.generic_inverted_index.len()..=generic_id {
                        serialized_index.generic_inverted_index.push(Vec::new());
                    }
                    &mut serialized_index.generic_inverted_index[generic_id]
                };
                while postings.len() <= search_type_size {
                    postings.push(Vec::new());
                }
                if postings[search_type_size].last() != Some(&(new_entry_id as u32)) {
                    postings[search_type_size].push(new_entry_id as u32);
                }
            }
        }
    }

    Ok(serialized_index.sort())
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
        clean::RawPointer { .. } => Some(RenderTypeId::Primitive(clean::PrimitiveType::RawPointer)),
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
        Type::BorrowedRef { lifetime: _, mutability, ref type_ }
        | Type::RawPointer(mutability, ref type_) => {
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
