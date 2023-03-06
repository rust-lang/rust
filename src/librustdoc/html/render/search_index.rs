use std::collections::hash_map::Entry;
use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use serde::ser::{Serialize, SerializeStruct, Serializer};

use crate::clean;
use crate::clean::types::{FnRetTy, Function, Generics, ItemId, Type, WherePredicate};
use crate::formats::cache::{Cache, OrphanImplItem};
use crate::formats::item_type::ItemType;
use crate::html::format::join_with_double_colon;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::{IndexItem, IndexItemFunctionType, RenderType, RenderTypeId};

/// Builds the search index from the collected metadata
pub(crate) fn build_index<'tcx>(
    krate: &clean::Crate,
    cache: &mut Cache,
    tcx: TyCtxt<'tcx>,
) -> String {
    let mut itemid_to_pathid = FxHashMap::default();
    let mut primitives = FxHashMap::default();
    let mut crate_paths = vec![];

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &OrphanImplItem { parent, ref item, ref impl_generics } in &cache.orphan_impl_items {
        if let Some((fqp, _)) = cache.paths.get(&parent) {
            let desc = item
                .doc_value()
                .map_or_else(String::new, |s| short_markdown_summary(&s, &item.link_names(cache)));
            cache.search_index.push(IndexItem {
                ty: item.type_(),
                name: item.name.unwrap(),
                path: join_with_double_colon(&fqp[..fqp.len() - 1]),
                desc,
                parent: Some(parent),
                parent_idx: None,
                search_type: get_function_type_for_search(item, tcx, impl_generics.as_ref(), cache),
                aliases: item.attrs.get_doc_aliases(),
            });
        }
    }

    let crate_doc = krate
        .module
        .doc_value()
        .map_or_else(String::new, |s| short_markdown_summary(&s, &krate.module.link_names(cache)));

    // Aliases added through `#[doc(alias = "...")]`. Since a few items can have the same alias,
    // we need the alias element to have an array of items.
    let mut aliases: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    // Sort search index items. This improves the compressibility of the search index.
    cache.search_index.sort_unstable_by(|k1, k2| {
        // `sort_unstable_by_key` produces lifetime errors
        let k1 = (&k1.path, k1.name.as_str(), &k1.ty, &k1.parent);
        let k2 = (&k2.path, k2.name.as_str(), &k2.ty, &k2.parent);
        std::cmp::Ord::cmp(&k1, &k2)
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
    let mut lastpathid = 0usize;

    // First, on function signatures
    let mut search_index = std::mem::replace(&mut cache.search_index, Vec::new());
    for item in search_index.iter_mut() {
        fn insert_into_map<F: std::hash::Hash + Eq>(
            ty: &mut RenderType,
            map: &mut FxHashMap<F, usize>,
            itemid: F,
            lastpathid: &mut usize,
            crate_paths: &mut Vec<(ItemType, Symbol)>,
            item_type: ItemType,
            path: Symbol,
        ) {
            match map.entry(itemid) {
                Entry::Occupied(entry) => ty.id = Some(RenderTypeId::Index(*entry.get())),
                Entry::Vacant(entry) => {
                    let pathid = *lastpathid;
                    entry.insert(pathid);
                    *lastpathid += 1;
                    crate_paths.push((item_type, path));
                    ty.id = Some(RenderTypeId::Index(pathid));
                }
            }
        }

        fn convert_render_type(
            ty: &mut RenderType,
            cache: &mut Cache,
            itemid_to_pathid: &mut FxHashMap<ItemId, usize>,
            primitives: &mut FxHashMap<Symbol, usize>,
            lastpathid: &mut usize,
            crate_paths: &mut Vec<(ItemType, Symbol)>,
        ) {
            if let Some(generics) = &mut ty.generics {
                for item in generics {
                    convert_render_type(
                        item,
                        cache,
                        itemid_to_pathid,
                        primitives,
                        lastpathid,
                        crate_paths,
                    );
                }
            }
            let Cache { ref paths, ref external_paths, .. } = *cache;
            let Some(id) = ty.id.clone() else {
                assert!(ty.generics.is_some());
                return;
            };
            match id {
                RenderTypeId::DefId(defid) => {
                    if let Some(&(ref fqp, item_type)) =
                        paths.get(&defid).or_else(|| external_paths.get(&defid))
                    {
                        insert_into_map(
                            ty,
                            itemid_to_pathid,
                            ItemId::DefId(defid),
                            lastpathid,
                            crate_paths,
                            item_type,
                            *fqp.last().unwrap(),
                        );
                    } else {
                        ty.id = None;
                    }
                }
                RenderTypeId::Primitive(primitive) => {
                    let sym = primitive.as_sym();
                    insert_into_map(
                        ty,
                        primitives,
                        sym,
                        lastpathid,
                        crate_paths,
                        ItemType::Primitive,
                        sym,
                    );
                }
                RenderTypeId::Index(_) => {}
            }
        }
        if let Some(search_type) = &mut item.search_type {
            for item in &mut search_type.inputs {
                convert_render_type(
                    item,
                    cache,
                    &mut itemid_to_pathid,
                    &mut primitives,
                    &mut lastpathid,
                    &mut crate_paths,
                );
            }
            for item in &mut search_type.output {
                convert_render_type(
                    item,
                    cache,
                    &mut itemid_to_pathid,
                    &mut primitives,
                    &mut lastpathid,
                    &mut crate_paths,
                );
            }
        }
    }

    let Cache { ref paths, .. } = *cache;

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
                            crate_paths.push((short, *fqp.last().unwrap()));
                            Some(pathid)
                        } else {
                            None
                        }
                    }
                });

            // Omit the parent path if it is same to that of the prior item.
            if lastpath == &item.path {
                item.path.clear();
            } else {
                lastpath = &item.path;
            }

            &*item
        })
        .collect();

    struct CrateData<'a> {
        doc: String,
        items: Vec<&'a IndexItem>,
        paths: Vec<(ItemType, Symbol)>,
        // The String is alias name and the vec is the list of the elements with this alias.
        //
        // To be noted: the `usize` elements are indexes to `items`.
        aliases: &'a BTreeMap<String, Vec<usize>>,
    }

    impl<'a> Serialize for CrateData<'a> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let has_aliases = !self.aliases.is_empty();
            let mut crate_data =
                serializer.serialize_struct("CrateData", if has_aliases { 9 } else { 8 })?;
            crate_data.serialize_field("doc", &self.doc)?;
            crate_data.serialize_field(
                "t",
                &self
                    .items
                    .iter()
                    .map(|item| {
                        let n = item.ty as u8;
                        let c = char::try_from(n + b'A').expect("item types must fit in ASCII");
                        assert!(c <= 'z', "item types must fit within ASCII printables");
                        c
                    })
                    .collect::<String>(),
            )?;
            crate_data.serialize_field(
                "n",
                &self.items.iter().map(|item| item.name.as_str()).collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "q",
                &self.items.iter().map(|item| &item.path).collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "d",
                &self.items.iter().map(|item| &item.desc).collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "i",
                &self
                    .items
                    .iter()
                    .map(|item| {
                        assert_eq!(
                            item.parent.is_some(),
                            item.parent_idx.is_some(),
                            "`{}` is missing idx",
                            item.name
                        );
                        // 0 is a sentinel, everything else is one-indexed
                        item.parent_idx.map(|x| x + 1).unwrap_or(0)
                    })
                    .collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "f",
                &self
                    .items
                    .iter()
                    .map(|item| {
                        // Fake option to get `0` out as a sentinel instead of `null`.
                        // We want to use `0` because it's three less bytes.
                        enum FunctionOption<'a> {
                            Function(&'a IndexItemFunctionType),
                            None,
                        }
                        impl<'a> Serialize for FunctionOption<'a> {
                            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                            where
                                S: Serializer,
                            {
                                match self {
                                    FunctionOption::None => 0.serialize(serializer),
                                    FunctionOption::Function(ty) => ty.serialize(serializer),
                                }
                            }
                        }
                        match &item.search_type {
                            Some(ty) => FunctionOption::Function(ty),
                            None => FunctionOption::None,
                        }
                    })
                    .collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "p",
                &self.paths.iter().map(|(it, s)| (it, s.as_str())).collect::<Vec<_>>(),
            )?;
            if has_aliases {
                crate_data.serialize_field("a", &self.aliases)?;
            }
            crate_data.end()
        }
    }

    // Collect the index into a string
    format!(
        r#""{}":{}"#,
        krate.name(tcx),
        serde_json::to_string(&CrateData {
            doc: crate_doc,
            items: crate_items,
            paths: crate_paths,
            aliases: &aliases,
        })
        .expect("failed serde conversion")
        // All these `replace` calls are because we have to go through JS string for JSON content.
        .replace('\\', r"\\")
        .replace('\'', r"\'")
        // We need to escape double quotes for the JSON.
        .replace("\\\"", "\\\\\"")
    )
}

pub(crate) fn get_function_type_for_search<'tcx>(
    item: &clean::Item,
    tcx: TyCtxt<'tcx>,
    impl_generics: Option<&(clean::Type, clean::Generics)>,
    cache: &Cache,
) -> Option<IndexItemFunctionType> {
    let (mut inputs, mut output) = match *item.kind {
        clean::FunctionItem(ref f) => get_fn_inputs_and_outputs(f, tcx, impl_generics, cache),
        clean::MethodItem(ref m, _) => get_fn_inputs_and_outputs(m, tcx, impl_generics, cache),
        clean::TyMethodItem(ref m) => get_fn_inputs_and_outputs(m, tcx, impl_generics, cache),
        _ => return None,
    };

    inputs.retain(|a| a.id.is_some() || a.generics.is_some());
    output.retain(|a| a.id.is_some() || a.generics.is_some());

    Some(IndexItemFunctionType { inputs, output })
}

fn get_index_type(clean_type: &clean::Type, generics: Vec<RenderType>) -> RenderType {
    RenderType {
        id: get_index_type_id(clean_type),
        generics: if generics.is_empty() { None } else { Some(generics) },
    }
}

fn get_index_type_id(clean_type: &clean::Type) -> Option<RenderTypeId> {
    match *clean_type {
        clean::Type::Path { ref path, .. } => Some(RenderTypeId::DefId(path.def_id())),
        clean::DynTrait(ref bounds, _) => {
            bounds.get(0).map(|b| RenderTypeId::DefId(b.trait_.def_id()))
        }
        clean::Primitive(p) => Some(RenderTypeId::Primitive(p)),
        clean::BorrowedRef { ref type_, .. } | clean::RawPointer(_, ref type_) => {
            get_index_type_id(type_)
        }
        clean::BareFunction(_)
        | clean::Generic(_)
        | clean::ImplTrait(_)
        | clean::Tuple(_)
        | clean::Slice(_)
        | clean::Array(_, _)
        | clean::QPath { .. }
        | clean::Infer => None,
    }
}

/// The point of this function is to replace bounds with types.
///
/// i.e. `[T, U]` when you have the following bounds: `T: Display, U: Option<T>` will return
/// `[Display, Option]`. If a type parameter has no trait bound, it is discarded.
///
/// Important note: It goes through generics recursively. So if you have
/// `T: Option<Result<(), ()>>`, it'll go into `Option` and then into `Result`.
#[instrument(level = "trace", skip(tcx, res, cache))]
fn add_generics_and_bounds_as_types<'tcx, 'a>(
    self_: Option<&'a Type>,
    generics: &Generics,
    arg: &'a Type,
    tcx: TyCtxt<'tcx>,
    recurse: usize,
    res: &mut Vec<RenderType>,
    cache: &Cache,
) {
    fn insert_ty(res: &mut Vec<RenderType>, ty: Type, mut generics: Vec<RenderType>) {
        // generics and impl trait are both identified by their generics,
        // rather than a type name itself
        let anonymous = ty.is_full_generic() || ty.is_impl_trait();
        let generics_empty = generics.is_empty();

        if anonymous {
            if generics_empty {
                // This is a type parameter with no trait bounds (for example: `T` in
                // `fn f<T>(p: T)`, so not useful for the rustdoc search because we would end up
                // with an empty type with an empty name. Let's just discard it.
                return;
            } else if generics.len() == 1 {
                // In this case, no need to go through an intermediate state if the type parameter
                // contains only one trait bound.
                //
                // For example:
                //
                // `fn foo<T: Display>(r: Option<T>) {}`
                //
                // In this case, it would contain:
                //
                // ```
                // [{
                //     name: "option",
                //     generics: [{
                //         name: "",
                //         generics: [
                //             name: "Display",
                //             generics: []
                //         }]
                //     }]
                // }]
                // ```
                //
                // After removing the intermediate (unnecessary) type parameter, it'll become:
                //
                // ```
                // [{
                //     name: "option",
                //     generics: [{
                //         name: "Display",
                //         generics: []
                //     }]
                // }]
                // ```
                //
                // To be noted that it can work if there is ONLY ONE trait bound, otherwise we still
                // need to keep it as is!
                res.push(generics.pop().unwrap());
                return;
            }
        }
        let index_ty = get_index_type(&ty, generics);
        if index_ty.id.is_none() && generics_empty {
            return;
        }
        res.push(index_ty);
    }

    if recurse >= 10 {
        // FIXME: remove this whole recurse thing when the recursion bug is fixed
        // See #59502 for the original issue.
        return;
    }

    // First, check if it's "Self".
    let arg = if let Some(self_) = self_ {
        match &*arg {
            Type::BorrowedRef { type_, .. } if type_.is_self_type() => self_,
            type_ if type_.is_self_type() => self_,
            arg => arg,
        }
    } else {
        arg
    };

    // If this argument is a type parameter and not a trait bound or a type, we need to look
    // for its bounds.
    if let Type::Generic(arg_s) = *arg {
        // First we check if the bounds are in a `where` predicate...
        if let Some(where_pred) = generics.where_predicates.iter().find(|g| match g {
            WherePredicate::BoundPredicate { ty: Type::Generic(ty_s), .. } => *ty_s == arg_s,
            _ => false,
        }) {
            let mut ty_generics = Vec::new();
            let bounds = where_pred.get_bounds().unwrap_or_else(|| &[]);
            for bound in bounds.iter() {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    add_generics_and_bounds_as_types(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_generics,
                        cache,
                    );
                }
            }
            insert_ty(res, arg.clone(), ty_generics);
        }
        // Otherwise we check if the trait bounds are "inlined" like `T: Option<u32>`...
        if let Some(bound) = generics.params.iter().find(|g| g.is_type() && g.name == arg_s) {
            let mut ty_generics = Vec::new();
            for bound in bound.get_bounds().unwrap_or(&[]) {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    add_generics_and_bounds_as_types(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_generics,
                        cache,
                    );
                }
            }
            insert_ty(res, arg.clone(), ty_generics);
        }
    } else if let Type::ImplTrait(ref bounds) = *arg {
        let mut ty_generics = Vec::new();
        for bound in bounds {
            if let Some(path) = bound.get_trait_path() {
                let ty = Type::Path { path };
                add_generics_and_bounds_as_types(
                    self_,
                    generics,
                    &ty,
                    tcx,
                    recurse + 1,
                    &mut ty_generics,
                    cache,
                );
            }
        }
        insert_ty(res, arg.clone(), ty_generics);
    } else {
        // This is not a type parameter. So for example if we have `T, U: Option<T>`, and we're
        // looking at `Option`, we enter this "else" condition, otherwise if it's `T`, we don't.
        //
        // So in here, we can add it directly and look for its own type parameters (so for `Option`,
        // we will look for them but not for `T`).
        let mut ty_generics = Vec::new();
        if let Some(arg_generics) = arg.generics() {
            for gen in arg_generics.iter() {
                add_generics_and_bounds_as_types(
                    self_,
                    generics,
                    gen,
                    tcx,
                    recurse + 1,
                    &mut ty_generics,
                    cache,
                );
            }
        }
        insert_ty(res, arg.clone(), ty_generics);
    }
}

/// Return the full list of types when bounds have been resolved.
///
/// i.e. `fn foo<A: Display, B: Option<A>>(x: u32, y: B)` will return
/// `[u32, Display, Option]`.
fn get_fn_inputs_and_outputs<'tcx>(
    func: &Function,
    tcx: TyCtxt<'tcx>,
    impl_generics: Option<&(clean::Type, clean::Generics)>,
    cache: &Cache,
) -> (Vec<RenderType>, Vec<RenderType>) {
    let decl = &func.decl;

    let combined_generics;
    let (self_, generics) = if let Some((impl_self, impl_generics)) = impl_generics {
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

    let mut all_types = Vec::new();
    for arg in decl.inputs.values.iter() {
        let mut args = Vec::new();
        add_generics_and_bounds_as_types(self_, generics, &arg.type_, tcx, 0, &mut args, cache);
        if !args.is_empty() {
            all_types.extend(args);
        } else {
            all_types.push(get_index_type(&arg.type_, vec![]));
        }
    }

    let mut ret_types = Vec::new();
    match decl.output {
        FnRetTy::Return(ref return_type) => {
            add_generics_and_bounds_as_types(
                self_,
                generics,
                return_type,
                tcx,
                0,
                &mut ret_types,
                cache,
            );
            if ret_types.is_empty() {
                ret_types.push(get_index_type(return_type, vec![]));
            }
        }
        _ => {}
    };
    (all_types, ret_types)
}
