use std::collections::hash_map::Entry;
use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::Symbol;
use serde::ser::{Serialize, SerializeStruct, Serializer};

use crate::clean;
use crate::clean::types::{FnRetTy, Function, GenericBound, Generics, Type, WherePredicate};
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::{IndexItem, IndexItemFunctionType, RenderType, TypeWithKind};

/// Builds the search index from the collected metadata
crate fn build_index<'tcx>(krate: &clean::Crate, cache: &mut Cache, tcx: TyCtxt<'tcx>) -> String {
    let mut defid_to_pathid = FxHashMap::default();
    let mut crate_paths = vec![];

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &(did, ref item) in &cache.orphan_impl_items {
        if let Some(&(ref fqp, _)) = cache.paths.get(&did) {
            let desc = item
                .doc_value()
                .map_or_else(String::new, |s| short_markdown_summary(&s, &item.link_names(cache)));
            cache.search_index.push(IndexItem {
                ty: item.type_(),
                name: item.name.unwrap().to_string(),
                path: fqp[..fqp.len() - 1].join("::"),
                desc,
                parent: Some(did),
                parent_idx: None,
                search_type: get_function_type_for_search(item, tcx),
                aliases: item.attrs.get_doc_aliases(),
            });
        }
    }

    let crate_doc = krate
        .module
        .doc_value()
        .map_or_else(String::new, |s| short_markdown_summary(&s, &krate.module.link_names(cache)));

    let Cache { ref mut search_index, ref paths, .. } = *cache;

    // Aliases added through `#[doc(alias = "...")]`. Since a few items can have the same alias,
    // we need the alias element to have an array of items.
    let mut aliases: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    // Sort search index items. This improves the compressibility of the search index.
    search_index.sort_unstable_by(|k1, k2| {
        // `sort_unstable_by_key` produces lifetime errors
        let k1 = (&k1.path, &k1.name, &k1.ty, &k1.parent);
        let k2 = (&k2.path, &k2.name, &k2.ty, &k2.parent);
        std::cmp::Ord::cmp(&k1, &k2)
    });

    // Set up alias indexes.
    for (i, item) in search_index.iter().enumerate() {
        for alias in &item.aliases[..] {
            aliases.entry(alias.as_str().to_lowercase()).or_default().push(i);
        }
    }

    // Reduce `DefId` in paths into smaller sequential numbers,
    // and prune the paths that do not appear in the index.
    let mut lastpath = "";
    let mut lastpathid = 0usize;

    let crate_items: Vec<&IndexItem> = search_index
        .iter_mut()
        .map(|item| {
            item.parent_idx = item.parent.and_then(|defid| match defid_to_pathid.entry(defid) {
                Entry::Occupied(entry) => Some(*entry.get()),
                Entry::Vacant(entry) => {
                    let pathid = lastpathid;
                    entry.insert(pathid);
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
        paths: Vec<(ItemType, String)>,
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
                &self.items.iter().map(|item| &item.ty).collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "n",
                &self.items.iter().map(|item| &item.name).collect::<Vec<_>>(),
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
                        item.parent_idx.map(|x| x + 1).unwrap_or(0)
                    })
                    .collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field(
                "f",
                &self.items.iter().map(|item| &item.search_type).collect::<Vec<_>>(),
            )?;
            crate_data.serialize_field("p", &self.paths)?;
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
        .replace(r#"\"#, r"\\")
        .replace(r#"'"#, r"\'")
        // We need to escape double quotes for the JSON.
        .replace("\\\"", "\\\\\"")
    )
}

crate fn get_function_type_for_search<'tcx>(
    item: &clean::Item,
    tcx: TyCtxt<'tcx>,
) -> Option<IndexItemFunctionType> {
    let (mut inputs, mut output) = match *item.kind {
        clean::FunctionItem(ref f) => get_fn_inputs_and_outputs(f, tcx),
        clean::MethodItem(ref m, _) => get_fn_inputs_and_outputs(m, tcx),
        clean::TyMethodItem(ref m) => get_fn_inputs_and_outputs(m, tcx),
        _ => return None,
    };

    inputs.retain(|a| a.ty.name.is_some());
    output.retain(|a| a.ty.name.is_some());

    Some(IndexItemFunctionType { inputs, output })
}

fn get_index_type(clean_type: &clean::Type, generics: Vec<TypeWithKind>) -> RenderType {
    RenderType {
        name: get_index_type_name(clean_type).map(|s| s.as_str().to_ascii_lowercase()),
        generics: if generics.is_empty() { None } else { Some(generics) },
    }
}

fn get_index_type_name(clean_type: &clean::Type) -> Option<Symbol> {
    match *clean_type {
        clean::Type::Path { ref path, .. } => {
            let path_segment = path.segments.last().unwrap();
            Some(path_segment.name)
        }
        clean::DynTrait(ref bounds, _) => {
            let path = &bounds[0].trait_;
            Some(path.segments.last().unwrap().name)
        }
        clean::Generic(s) => Some(s),
        clean::Primitive(ref p) => Some(p.as_sym()),
        clean::BorrowedRef { ref type_, .. } => get_index_type_name(type_),
        clean::BareFunction(_)
        | clean::Tuple(_)
        | clean::Slice(_)
        | clean::Array(_, _)
        | clean::RawPointer(_, _)
        | clean::QPath { .. }
        | clean::Infer
        | clean::ImplTrait(_) => None,
    }
}

/// The point of this function is to replace bounds with types.
///
/// i.e. `[T, U]` when you have the following bounds: `T: Display, U: Option<T>` will return
/// `[Display, Option]`. If a type parameter has no trait bound, it is discarded.
///
/// Important note: It goes through generics recursively. So if you have
/// `T: Option<Result<(), ()>>`, it'll go into `Option` and then into `Result`.
#[instrument(level = "trace", skip(tcx, res))]
fn add_generics_and_bounds_as_types<'tcx>(
    generics: &Generics,
    arg: &Type,
    tcx: TyCtxt<'tcx>,
    recurse: usize,
    res: &mut Vec<TypeWithKind>,
) {
    fn insert_ty(
        res: &mut Vec<TypeWithKind>,
        tcx: TyCtxt<'_>,
        ty: Type,
        mut generics: Vec<TypeWithKind>,
    ) {
        let is_full_generic = ty.is_full_generic();

        if is_full_generic {
            if generics.is_empty() {
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
        let mut index_ty = get_index_type(&ty, generics);
        if index_ty.name.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
            return;
        }
        if is_full_generic {
            // We remove the name of the full generic because we have no use for it.
            index_ty.name = Some(String::new());
            res.push(TypeWithKind::from((index_ty, ItemType::Generic)));
        } else if let Some(kind) = ty.def_id_no_primitives().map(|did| tcx.def_kind(did).into()) {
            res.push(TypeWithKind::from((index_ty, kind)));
        } else if ty.is_primitive() {
            // This is a primitive, let's store it as such.
            res.push(TypeWithKind::from((index_ty, ItemType::Primitive)));
        }
    }

    if recurse >= 10 {
        // FIXME: remove this whole recurse thing when the recursion bug is fixed
        // See #59502 for the original issue.
        return;
    }

    // If this argument is a type parameter and not a trait bound or a type, we need to look
    // for its bounds.
    if let Type::Generic(arg_s) = *arg {
        // First we check if the bounds are in a `where` predicate...
        if let Some(where_pred) = generics.where_predicates.iter().find(|g| match g {
            WherePredicate::BoundPredicate { ty, .. } => {
                ty.def_id_no_primitives() == arg.def_id_no_primitives()
            }
            _ => false,
        }) {
            let mut ty_generics = Vec::new();
            let bounds = where_pred.get_bounds().unwrap_or_else(|| &[]);
            for bound in bounds.iter() {
                if let GenericBound::TraitBound(poly_trait, _) = bound {
                    for param_def in poly_trait.generic_params.iter() {
                        match &param_def.kind {
                            clean::GenericParamDefKind::Type { default: Some(ty), .. } => {
                                add_generics_and_bounds_as_types(
                                    generics,
                                    ty,
                                    tcx,
                                    recurse + 1,
                                    &mut ty_generics,
                                )
                            }
                            _ => {}
                        }
                    }
                }
            }
            insert_ty(res, tcx, arg.clone(), ty_generics);
        }
        // Otherwise we check if the trait bounds are "inlined" like `T: Option<u32>`...
        if let Some(bound) = generics.params.iter().find(|g| g.is_type() && g.name == arg_s) {
            let mut ty_generics = Vec::new();
            for bound in bound.get_bounds().unwrap_or(&[]) {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    add_generics_and_bounds_as_types(
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_generics,
                    );
                }
            }
            insert_ty(res, tcx, arg.clone(), ty_generics);
        }
    } else {
        // This is not a type parameter. So for example if we have `T, U: Option<T>`, and we're
        // looking at `Option`, we enter this "else" condition, otherwise if it's `T`, we don't.
        //
        // So in here, we can add it directly and look for its own type parameters (so for `Option`,
        // we will look for them but not for `T`).
        let mut ty_generics = Vec::new();
        if let Some(arg_generics) = arg.generics() {
            for gen in arg_generics.iter() {
                add_generics_and_bounds_as_types(generics, gen, tcx, recurse + 1, &mut ty_generics);
            }
        }
        insert_ty(res, tcx, arg.clone(), ty_generics);
    }
}

/// Return the full list of types when bounds have been resolved.
///
/// i.e. `fn foo<A: Display, B: Option<A>>(x: u32, y: B)` will return
/// `[u32, Display, Option]`.
fn get_fn_inputs_and_outputs<'tcx>(
    func: &Function,
    tcx: TyCtxt<'tcx>,
) -> (Vec<TypeWithKind>, Vec<TypeWithKind>) {
    let decl = &func.decl;
    let generics = &func.generics;

    let mut all_types = Vec::new();
    for arg in decl.inputs.values.iter() {
        if arg.type_.is_self_type() {
            continue;
        }
        let mut args = Vec::new();
        add_generics_and_bounds_as_types(generics, &arg.type_, tcx, 0, &mut args);
        if !args.is_empty() {
            all_types.extend(args);
        } else {
            if let Some(kind) = arg.type_.def_id_no_primitives().map(|did| tcx.def_kind(did).into())
            {
                all_types.push(TypeWithKind::from((get_index_type(&arg.type_, vec![]), kind)));
            }
        }
    }

    let mut ret_types = Vec::new();
    match decl.output {
        FnRetTy::Return(ref return_type) => {
            add_generics_and_bounds_as_types(generics, return_type, tcx, 0, &mut ret_types);
            if ret_types.is_empty() {
                if let Some(kind) =
                    return_type.def_id_no_primitives().map(|did| tcx.def_kind(did).into())
                {
                    ret_types.push(TypeWithKind::from((get_index_type(return_type, vec![]), kind)));
                }
            }
        }
        _ => {}
    };
    (all_types, ret_types)
}
