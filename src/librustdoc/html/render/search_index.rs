use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, VecDeque};

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::sym;
use rustc_span::symbol::Symbol;
use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};
use thin_vec::ThinVec;

use crate::clean;
use crate::clean::types::{Function, Generics, ItemId, Type, WherePredicate};
use crate::formats::cache::{Cache, OrphanImplItem};
use crate::formats::item_type::ItemType;
use crate::html::format::join_with_double_colon;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::{self, IndexItem, IndexItemFunctionType, RenderType, RenderTypeId};

/// Builds the search index from the collected metadata
pub(crate) fn build_index<'tcx>(
    krate: &clean::Crate,
    cache: &mut Cache,
    tcx: TyCtxt<'tcx>,
) -> String {
    let mut itemid_to_pathid = FxHashMap::default();
    let mut primitives = FxHashMap::default();
    let mut associated_types = FxHashMap::default();
    let mut crate_paths = vec![];

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &OrphanImplItem { impl_id, parent, ref item, ref impl_generics } in &cache.orphan_impl_items
    {
        if let Some((fqp, _)) = cache.paths.get(&parent) {
            let desc = short_markdown_summary(&item.doc_value(), &item.link_names(cache));
            cache.search_index.push(IndexItem {
                ty: item.type_(),
                name: item.name.unwrap(),
                path: join_with_double_colon(&fqp[..fqp.len() - 1]),
                desc,
                parent: Some(parent),
                parent_idx: None,
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
        let k1 = (&k1.path, k1.name.as_str(), &k1.ty, &k1.parent);
        let k2 = (&k2.path, k2.name.as_str(), &k2.ty, &k2.parent);
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
    let mut search_index = std::mem::replace(&mut cache.search_index, Vec::new());
    for item in search_index.iter_mut() {
        fn insert_into_map<F: std::hash::Hash + Eq>(
            map: &mut FxHashMap<F, isize>,
            itemid: F,
            lastpathid: &mut isize,
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>)>,
            item_type: ItemType,
            path: &[Symbol],
        ) -> RenderTypeId {
            match map.entry(itemid) {
                Entry::Occupied(entry) => RenderTypeId::Index(*entry.get()),
                Entry::Vacant(entry) => {
                    let pathid = *lastpathid;
                    entry.insert(pathid);
                    *lastpathid += 1;
                    crate_paths.push((item_type, path.to_vec()));
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
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>)>,
        ) -> Option<RenderTypeId> {
            let Cache { ref paths, ref external_paths, .. } = *cache;
            match id {
                RenderTypeId::DefId(defid) => {
                    if let Some(&(ref fqp, item_type)) =
                        paths.get(&defid).or_else(|| external_paths.get(&defid))
                    {
                        Some(insert_into_map(
                            itemid_to_pathid,
                            ItemId::DefId(defid),
                            lastpathid,
                            crate_paths,
                            item_type,
                            fqp,
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
            crate_paths: &mut Vec<(ItemType, Vec<Symbol>)>,
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
                        );
                    }
                    true
                });
            }
            let Some(id) = ty.id.clone() else {
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
                    );
                }
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
                            crate_paths.push((short, fqp.clone()));
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
        doc: String,
        items: Vec<&'a IndexItem>,
        paths: Vec<(ItemType, Vec<Symbol>)>,
        // The String is alias name and the vec is the list of the elements with this alias.
        //
        // To be noted: the `usize` elements are indexes to `items`.
        aliases: &'a BTreeMap<String, Vec<usize>>,
        // Used when a type has more than one impl with an associated item with the same name.
        associated_item_disambiguators: &'a Vec<(usize, String)>,
    }

    struct Paths {
        ty: ItemType,
        name: Symbol,
        path: Option<usize>,
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
            seq.end()
        }
    }

    impl<'a> Serialize for CrateData<'a> {
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
            for (ty, path) in &self.paths {
                if path.len() < 2 {
                    paths.push(Paths { ty: *ty, name: path[0], path: None });
                    continue;
                }
                let full_path = join_with_double_colon(&path[..path.len() - 1]);
                if let Some(index) = mod_paths.get(&full_path) {
                    paths.push(Paths { ty: *ty, name: *path.last().unwrap(), path: Some(*index) });
                    continue;
                }
                // It means it comes from an external crate so the item and its path will be
                // stored into another array.
                //
                // `index` is put after the last `mod_paths`
                let index = extra_paths.len() + self.items.len();
                if !revert_extra_paths.contains_key(&index) {
                    revert_extra_paths.insert(index, full_path.clone());
                }
                match extra_paths.entry(full_path) {
                    Entry::Occupied(entry) => {
                        paths.push(Paths {
                            ty: *ty,
                            name: *path.last().unwrap(),
                            path: Some(*entry.get()),
                        });
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(index);
                        paths.push(Paths {
                            ty: *ty,
                            name: *path.last().unwrap(),
                            path: Some(index),
                        });
                    }
                }
            }

            let mut names = Vec::with_capacity(self.items.len());
            let mut types = String::with_capacity(self.items.len());
            let mut full_paths = Vec::with_capacity(self.items.len());
            let mut descriptions = Vec::with_capacity(self.items.len());
            let mut parents = Vec::with_capacity(self.items.len());
            let mut functions = String::with_capacity(self.items.len());
            let mut deprecated = Vec::with_capacity(self.items.len());

            let mut backref_queue = VecDeque::new();

            for (index, item) in self.items.iter().enumerate() {
                let n = item.ty as u8;
                let c = char::try_from(n + b'A').expect("item types must fit in ASCII");
                assert!(c <= 'z', "item types must fit within ASCII printables");
                types.push(c);

                assert_eq!(
                    item.parent.is_some(),
                    item.parent_idx.is_some(),
                    "`{}` is missing idx",
                    item.name
                );
                // 0 is a sentinel, everything else is one-indexed
                parents.push(item.parent_idx.map(|x| x + 1).unwrap_or(0));

                names.push(item.name.as_str());
                descriptions.push(&item.desc);

                if !item.path.is_empty() {
                    full_paths.push((index, &item.path));
                }

                match &item.search_type {
                    Some(ty) => ty.write_to_string(&mut functions, &mut backref_queue),
                    None => functions.push('`'),
                }

                if item.deprecation.is_some() {
                    deprecated.push(index);
                }
            }

            for (index, path) in &revert_extra_paths {
                full_paths.push((*index, path));
            }

            let has_aliases = !self.aliases.is_empty();
            let mut crate_data =
                serializer.serialize_struct("CrateData", if has_aliases { 9 } else { 8 })?;
            crate_data.serialize_field("doc", &self.doc)?;
            crate_data.serialize_field("t", &types)?;
            crate_data.serialize_field("n", &names)?;
            // Serialize as an array of item indices and full paths
            crate_data.serialize_field("q", &full_paths)?;
            crate_data.serialize_field("d", &descriptions)?;
            crate_data.serialize_field("i", &parents)?;
            crate_data.serialize_field("f", &functions)?;
            crate_data.serialize_field("c", &deprecated)?;
            crate_data.serialize_field("p", &paths)?;
            crate_data.serialize_field("b", &self.associated_item_disambiguators)?;
            if has_aliases {
                crate_data.serialize_field("a", &self.aliases)?;
            }
            crate_data.end()
        }
    }

    // Collect the index into a string
    format!(
        r#"["{}",{}]"#,
        krate.name(tcx),
        serde_json::to_string(&CrateData {
            doc: crate_doc,
            items: crate_items,
            paths: crate_paths,
            aliases: &aliases,
            associated_item_disambiguators: &associated_item_disambiguators,
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
                            args: Vec::new().into_boxed_slice(),
                            bindings: ThinVec::new(),
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
    let (mut inputs, mut output, where_clause) = match *item.kind {
        clean::FunctionItem(ref f) | clean::MethodItem(ref f, _) | clean::TyMethodItem(ref f) => {
            get_fn_inputs_and_outputs(f, tcx, impl_or_trait_generics, cache)
        }
        _ => return None,
    };

    inputs.retain(|a| a.id.is_some() || a.generics.is_some());
    output.retain(|a| a.id.is_some() || a.generics.is_some());

    Some(IndexItemFunctionType { inputs, output, where_clause })
}

fn get_index_type(
    clean_type: &clean::Type,
    generics: Vec<RenderType>,
    rgen: &mut FxHashMap<SimplifiedParam, (isize, Vec<RenderType>)>,
) -> RenderType {
    RenderType {
        id: get_index_type_id(clean_type, rgen),
        generics: if generics.is_empty() { None } else { Some(generics) },
        bindings: None,
    }
}

fn get_index_type_id(
    clean_type: &clean::Type,
    rgen: &mut FxHashMap<SimplifiedParam, (isize, Vec<RenderType>)>,
) -> Option<RenderTypeId> {
    use rustc_hir::def::{DefKind, Res};
    match *clean_type {
        clean::Type::Path { ref path, .. } => Some(RenderTypeId::DefId(path.def_id())),
        clean::DynTrait(ref bounds, _) => {
            bounds.get(0).map(|b| RenderTypeId::DefId(b.trait_.def_id()))
        }
        clean::Primitive(p) => Some(RenderTypeId::Primitive(p)),
        clean::BorrowedRef { ref type_, .. } | clean::RawPointer(_, ref type_) => {
            get_index_type_id(type_, rgen)
        }
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
        clean::Generic(_) | clean::ImplTrait(_) | clean::Infer => None,
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
fn simplify_fn_type<'tcx, 'a>(
    self_: Option<&'a Type>,
    generics: &Generics,
    arg: &'a Type,
    tcx: TyCtxt<'tcx>,
    recurse: usize,
    res: &mut Vec<RenderType>,
    rgen: &mut FxHashMap<SimplifiedParam, (isize, Vec<RenderType>)>,
    is_return: bool,
    cache: &Cache,
) {
    if recurse >= 10 {
        // FIXME: remove this whole recurse thing when the recursion bug is fixed
        // See #59502 for the original issue.
        return;
    }

    // First, check if it's "Self".
    let mut is_self = false;
    let mut arg = if let Some(self_) = self_ {
        match &*arg {
            Type::BorrowedRef { type_, .. } if type_.is_self_type() => {
                is_self = true;
                self_
            }
            type_ if type_.is_self_type() => {
                is_self = true;
                self_
            }
            arg => arg,
        }
    } else {
        arg
    };

    // strip references from the argument type
    while let Type::BorrowedRef { type_, .. } = &*arg {
        arg = &*type_;
    }

    // If this argument is a type parameter and not a trait bound or a type, we need to look
    // for its bounds.
    if let Type::Generic(arg_s) = *arg {
        // First we check if the bounds are in a `where` predicate...
        let mut type_bounds = Vec::new();
        for where_pred in generics.where_predicates.iter().filter(|g| match g {
            WherePredicate::BoundPredicate { ty: Type::Generic(ty_s), .. } => *ty_s == arg_s,
            _ => false,
        }) {
            let bounds = where_pred.get_bounds().unwrap_or_else(|| &[]);
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
    } else if let Type::ImplTrait(ref bounds) = *arg {
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
            // In parameter position, `impl Trait` is a unique thing.
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
    } else if let Type::Slice(ref ty) = *arg {
        let mut ty_generics = Vec::new();
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
        res.push(get_index_type(arg, ty_generics, rgen));
    } else if let Type::Array(ref ty, _) = *arg {
        let mut ty_generics = Vec::new();
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
        res.push(get_index_type(arg, ty_generics, rgen));
    } else if let Type::Tuple(ref tys) = *arg {
        let mut ty_generics = Vec::new();
        for ty in tys {
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
        res.push(get_index_type(arg, ty_generics, rgen));
    } else if let Type::BareFunction(ref bf) = *arg {
        let mut ty_generics = Vec::new();
        for ty in bf.decl.inputs.values.iter().map(|arg| &arg.type_) {
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
            id: get_index_type_id(&arg, rgen),
            bindings: Some(ty_bindings),
            generics: Some(ty_generics),
        });
    } else {
        // This is not a type parameter. So for example if we have `T, U: Option<T>`, and we're
        // looking at `Option`, we enter this "else" condition, otherwise if it's `T`, we don't.
        //
        // So in here, we can add it directly and look for its own type parameters (so for `Option`,
        // we will look for them but not for `T`).
        let mut ty_generics = Vec::new();
        let mut ty_bindings = Vec::new();
        if let Some(arg_generics) = arg.generic_args() {
            for ty in arg_generics.into_iter().filter_map(|gen| match gen {
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
            for binding in arg_generics.bindings() {
                simplify_fn_binding(
                    self_,
                    generics,
                    &binding,
                    tcx,
                    recurse + 1,
                    &mut ty_bindings,
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
            && trait_.items.iter().any(|at| at.is_ty_associated_type())
        {
            for assoc_ty in &trait_.items {
                if let clean::ItemKind::TyAssocTypeItem(_generics, bounds) = &*assoc_ty.kind
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
                    ty_bindings.push((
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
        let id = get_index_type_id(&arg, rgen);
        if id.is_some() || !ty_generics.is_empty() {
            res.push(RenderType {
                id,
                bindings: if ty_bindings.is_empty() { None } else { Some(ty_bindings) },
                generics: if ty_generics.is_empty() { None } else { Some(ty_generics) },
            });
        }
    }
}

fn simplify_fn_binding<'tcx, 'a>(
    self_: Option<&'a Type>,
    generics: &Generics,
    binding: &'a clean::TypeBinding,
    tcx: TyCtxt<'tcx>,
    recurse: usize,
    res: &mut Vec<(RenderTypeId, Vec<RenderType>)>,
    rgen: &mut FxHashMap<SimplifiedParam, (isize, Vec<RenderType>)>,
    is_return: bool,
    cache: &Cache,
) {
    let mut ty_binding_constraints = Vec::new();
    let ty_binding_assoc = RenderTypeId::AssociatedType(binding.assoc.name);
    for gen in &binding.assoc.args {
        match gen {
            clean::GenericArg::Type(arg) => simplify_fn_type(
                self_,
                generics,
                &arg,
                tcx,
                recurse + 1,
                &mut ty_binding_constraints,
                rgen,
                is_return,
                cache,
            ),
            clean::GenericArg::Lifetime(_)
            | clean::GenericArg::Const(_)
            | clean::GenericArg::Infer => {}
        }
    }
    for binding in binding.assoc.args.bindings() {
        simplify_fn_binding(
            self_,
            generics,
            &binding,
            tcx,
            recurse + 1,
            res,
            rgen,
            is_return,
            cache,
        );
    }
    match &binding.kind {
        clean::TypeBindingKind::Equality { term } => {
            if let clean::Term::Type(arg) = &term {
                simplify_fn_type(
                    self_,
                    generics,
                    arg,
                    tcx,
                    recurse + 1,
                    &mut ty_binding_constraints,
                    rgen,
                    is_return,
                    cache,
                );
            }
        }
        clean::TypeBindingKind::Constraint { bounds } => {
            for bound in &bounds[..] {
                if let Some(path) = bound.get_trait_path() {
                    let ty = Type::Path { path };
                    simplify_fn_type(
                        self_,
                        generics,
                        &ty,
                        tcx,
                        recurse + 1,
                        &mut ty_binding_constraints,
                        rgen,
                        is_return,
                        cache,
                    );
                }
            }
        }
    }
    res.push((ty_binding_assoc, ty_binding_constraints));
}

/// Return the full list of types when bounds have been resolved.
///
/// i.e. `fn foo<A: Display, B: Option<A>>(x: u32, y: B)` will return
/// `[u32, Display, Option]`.
fn get_fn_inputs_and_outputs<'tcx>(
    func: &Function,
    tcx: TyCtxt<'tcx>,
    impl_or_trait_generics: Option<&(clean::Type, clean::Generics)>,
    cache: &Cache,
) -> (Vec<RenderType>, Vec<RenderType>, Vec<Vec<RenderType>>) {
    let decl = &func.decl;

    let mut rgen: FxHashMap<SimplifiedParam, (isize, Vec<RenderType>)> = Default::default();

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

    let mut arg_types = Vec::new();
    for arg in decl.inputs.values.iter() {
        simplify_fn_type(
            self_,
            generics,
            &arg.type_,
            tcx,
            0,
            &mut arg_types,
            &mut rgen,
            false,
            cache,
        );
    }

    let mut ret_types = Vec::new();
    simplify_fn_type(self_, generics, &decl.output, tcx, 0, &mut ret_types, &mut rgen, true, cache);

    let mut simplified_params = rgen.into_values().collect::<Vec<_>>();
    simplified_params.sort_by_key(|(idx, _)| -idx);
    (arg_types, ret_types, simplified_params.into_iter().map(|(_idx, traits)| traits).collect())
}
