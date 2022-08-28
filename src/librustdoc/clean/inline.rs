//! Support for inlining external documentation into the current AST.

use std::iter::once;
use std::sync::Arc;

use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_metadata::creader::{CStore, LoadedMacro};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{kw, sym, Symbol};

use crate::clean::{
    self, clean_fn_decl_from_did_and_sig, clean_generics, clean_impl_item, clean_middle_assoc_item,
    clean_middle_field, clean_middle_ty, clean_trait_ref_with_bindings, clean_ty,
    clean_ty_generics, clean_variant_def, clean_visibility, utils, Attributes, AttributesExt,
    ImplKind, ItemId, Type, Visibility,
};
use crate::core::DocContext;
use crate::formats::item_type::ItemType;

/// Attempt to inline a definition into this AST.
///
/// This function will fetch the definition specified, and if it is
/// from another crate it will attempt to inline the documentation
/// from the other crate into this crate.
///
/// This is primarily used for `pub use` statements which are, in general,
/// implementation details. Inlining the documentation should help provide a
/// better experience when reading the documentation in this use case.
///
/// The returned value is `None` if the definition could not be inlined,
/// and `Some` of a vector of items if it was successfully expanded.
///
/// `parent_module` refers to the parent of the *re-export*, not the original item.
pub(crate) fn try_inline(
    cx: &mut DocContext<'_>,
    parent_module: DefId,
    import_def_id: Option<DefId>,
    res: Res,
    name: Symbol,
    attrs: Option<&[ast::Attribute]>,
    visited: &mut FxHashSet<DefId>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }
    let mut ret = Vec::new();

    debug!("attrs={:?}", attrs);
    let attrs_clone = attrs;

    let kind = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, ItemType::Trait);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::TraitItem(Box::new(build_external_trait(cx, did)))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, ItemType::Function);
            clean::FunctionItem(build_external_function(cx, did))
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, ItemType::Struct);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::StructItem(build_struct(cx, did))
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, ItemType::Union);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::UnionItem(build_union(cx, did))
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, ItemType::Typedef);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::TypedefItem(build_type_alias(cx, did))
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, ItemType::Enum);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::EnumItem(build_enum(cx, did))
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, ItemType::ForeignType);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::ForeignTypeItem
        }
        // Never inline enum variants but leave them shown as re-exports.
        Res::Def(DefKind::Variant, _) => return None,
        // Assume that enum variants and struct types are re-exported next to
        // their constructors.
        Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) => return Some(Vec::new()),
        Res::Def(DefKind::Mod, did) => {
            record_extern_fqn(cx, did, ItemType::Module);
            clean::ModuleItem(build_module(cx, did, visited))
        }
        Res::Def(DefKind::Static(_), did) => {
            record_extern_fqn(cx, did, ItemType::Static);
            clean::StaticItem(build_static(cx, did, cx.tcx.is_mutable_static(did)))
        }
        Res::Def(DefKind::Const, did) => {
            record_extern_fqn(cx, did, ItemType::Constant);
            clean::ConstantItem(build_const(cx, did))
        }
        Res::Def(DefKind::Macro(kind), did) => {
            let mac = build_macro(cx, did, name, import_def_id);

            let type_kind = match kind {
                MacroKind::Bang => ItemType::Macro,
                MacroKind::Attr => ItemType::ProcAttribute,
                MacroKind::Derive => ItemType::ProcDerive,
            };
            record_extern_fqn(cx, did, type_kind);
            mac
        }
        _ => return None,
    };

    let (attrs, cfg) = merge_attrs(cx, Some(parent_module), load_attrs(cx, did), attrs_clone);
    cx.inlined.insert(did.into());
    let mut item = clean::Item::from_def_id_and_attrs_and_parts(
        did,
        Some(name),
        kind,
        Box::new(attrs),
        cx,
        cfg,
    );
    if let Some(import_def_id) = import_def_id {
        // The visibility needs to reflect the one from the reexport and not from the "source" DefId.
        item.visibility = clean_visibility(cx.tcx.visibility(import_def_id));
    }
    ret.push(item);
    Some(ret)
}

pub(crate) fn try_inline_glob(
    cx: &mut DocContext<'_>,
    res: Res,
    visited: &mut FxHashSet<DefId>,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }

    match res {
        Res::Def(DefKind::Mod, did) => {
            let mut items = build_module_items(cx, did, visited, inlined_names);
            items.drain_filter(|item| {
                if let Some(name) = item.name {
                    // If an item with the same type and name already exists,
                    // it takes priority over the inlined stuff.
                    !inlined_names.insert((item.type_(), name))
                } else {
                    false
                }
            });
            Some(items)
        }
        // glob imports on things like enums aren't inlined even for local exports, so just bail
        _ => None,
    }
}

pub(crate) fn load_attrs<'hir>(cx: &DocContext<'hir>, did: DefId) -> &'hir [ast::Attribute] {
    cx.tcx.get_attrs_unchecked(did)
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub(crate) fn record_extern_fqn(cx: &mut DocContext<'_>, did: DefId, kind: ItemType) {
    let crate_name = cx.tcx.crate_name(did.krate);

    let relative =
        cx.tcx.def_path(did).data.into_iter().filter_map(|elem| elem.data.get_opt_name());
    let fqn = if let ItemType::Macro = kind {
        // Check to see if it is a macro 2.0 or built-in macro
        if matches!(
            CStore::from_tcx(cx.tcx).load_macro_untracked(did, cx.sess()),
            LoadedMacro::MacroDef(def, _)
                if matches!(&def.kind, ast::ItemKind::MacroDef(ast_def)
                    if !ast_def.macro_rules)
        ) {
            once(crate_name).chain(relative).collect()
        } else {
            vec![crate_name, relative.last().expect("relative was empty")]
        }
    } else {
        once(crate_name).chain(relative).collect()
    };

    if did.is_local() {
        cx.cache.exact_paths.insert(did, fqn);
    } else {
        cx.cache.external_paths.insert(did, (fqn, kind));
    }
}

pub(crate) fn build_external_trait(cx: &mut DocContext<'_>, did: DefId) -> clean::Trait {
    let trait_items = cx
        .tcx
        .associated_items(did)
        .in_definition_order()
        .map(|item| {
            // When building an external trait, the cleaned trait will have all items public,
            // which causes methods to have a `pub` prefix, which is invalid since items in traits
            // can not have a visibility prefix. Thus we override the visibility here manually.
            // See https://github.com/rust-lang/rust/issues/81274
            clean::Item { visibility: Visibility::Inherited, ..clean_middle_assoc_item(item, cx) }
        })
        .collect();

    let predicates = cx.tcx.predicates_of(did);
    let generics = clean_ty_generics(cx, cx.tcx.generics_of(did), predicates);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    clean::Trait { def_id: did, generics, items: trait_items, bounds: supertrait_bounds }
}

fn build_external_function<'tcx>(cx: &mut DocContext<'tcx>, did: DefId) -> Box<clean::Function> {
    let sig = cx.tcx.fn_sig(did);

    let predicates = cx.tcx.predicates_of(did);
    let (generics, decl) = clean::enter_impl_trait(cx, |cx| {
        // NOTE: generics need to be cleaned before the decl!
        let generics = clean_ty_generics(cx, cx.tcx.generics_of(did), predicates);
        let decl = clean_fn_decl_from_did_and_sig(cx, Some(did), sig);
        (generics, decl)
    });
    Box::new(clean::Function { decl, generics })
}

fn build_enum(cx: &mut DocContext<'_>, did: DefId) -> clean::Enum {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Enum {
        generics: clean_ty_generics(cx, cx.tcx.generics_of(did), predicates),
        variants: cx.tcx.adt_def(did).variants().iter().map(|v| clean_variant_def(v, cx)).collect(),
    }
}

fn build_struct(cx: &mut DocContext<'_>, did: DefId) -> clean::Struct {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Struct {
        struct_type: variant.ctor_kind,
        generics: clean_ty_generics(cx, cx.tcx.generics_of(did), predicates),
        fields: variant.fields.iter().map(|x| clean_middle_field(x, cx)).collect(),
    }
}

fn build_union(cx: &mut DocContext<'_>, did: DefId) -> clean::Union {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    let generics = clean_ty_generics(cx, cx.tcx.generics_of(did), predicates);
    let fields = variant.fields.iter().map(|x| clean_middle_field(x, cx)).collect();
    clean::Union { generics, fields }
}

fn build_type_alias(cx: &mut DocContext<'_>, did: DefId) -> Box<clean::Typedef> {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let type_ = clean_middle_ty(cx.tcx.type_of(did), cx, Some(did));

    Box::new(clean::Typedef {
        type_,
        generics: clean_ty_generics(cx, cx.tcx.generics_of(did), predicates),
        item_type: None,
    })
}

/// Builds all inherent implementations of an ADT (struct/union/enum) or Trait item/path/reexport.
pub(crate) fn build_impls(
    cx: &mut DocContext<'_>,
    parent_module: Option<DefId>,
    did: DefId,
    attrs: Option<&[ast::Attribute]>,
    ret: &mut Vec<clean::Item>,
) {
    let _prof_timer = cx.tcx.sess.prof.generic_activity("build_inherent_impls");
    let tcx = cx.tcx;

    // for each implementation of an item represented by `did`, build the clean::Item for that impl
    for &did in tcx.inherent_impls(did).iter() {
        build_impl(cx, parent_module, did, attrs, ret);
    }
}

/// `parent_module` refers to the parent of the re-export, not the original item
pub(crate) fn merge_attrs(
    cx: &mut DocContext<'_>,
    parent_module: Option<DefId>,
    old_attrs: &[ast::Attribute],
    new_attrs: Option<&[ast::Attribute]>,
) -> (clean::Attributes, Option<Arc<clean::cfg::Cfg>>) {
    // NOTE: If we have additional attributes (from a re-export),
    // always insert them first. This ensure that re-export
    // doc comments show up before the original doc comments
    // when we render them.
    if let Some(inner) = new_attrs {
        let mut both = inner.to_vec();
        both.extend_from_slice(old_attrs);
        (
            if let Some(new_id) = parent_module {
                Attributes::from_ast_with_additional(old_attrs, (inner, new_id))
            } else {
                Attributes::from_ast(&both)
            },
            both.cfg(cx.tcx, &cx.cache.hidden_cfg),
        )
    } else {
        (Attributes::from_ast(&old_attrs), old_attrs.cfg(cx.tcx, &cx.cache.hidden_cfg))
    }
}

/// Inline an `impl`, inherent or of a trait. The `did` must be for an `impl`.
pub(crate) fn build_impl(
    cx: &mut DocContext<'_>,
    parent_module: Option<DefId>,
    did: DefId,
    attrs: Option<&[ast::Attribute]>,
    ret: &mut Vec<clean::Item>,
) {
    if !cx.inlined.insert(did.into()) {
        return;
    }

    let _prof_timer = cx.tcx.sess.prof.generic_activity("build_impl");

    let tcx = cx.tcx;
    let associated_trait = tcx.impl_trait_ref(did);

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(traitref) = associated_trait {
            let did = traitref.def_id;
            if !cx.cache.access_levels.is_public(did) {
                return;
            }

            if let Some(stab) = tcx.lookup_stability(did) {
                if stab.is_unstable() && stab.feature == sym::rustc_private {
                    return;
                }
            }
        }
    }

    let impl_item = match did.as_local() {
        Some(did) => match &tcx.hir().expect_item(did).kind {
            hir::ItemKind::Impl(impl_) => Some(impl_),
            _ => panic!("`DefID` passed to `build_impl` is not an `impl"),
        },
        None => None,
    };

    let for_ = match &impl_item {
        Some(impl_) => clean_ty(impl_.self_ty, cx),
        None => clean_middle_ty(tcx.type_of(did), cx, Some(did)),
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(did) = for_.def_id(&cx.cache) {
            if !cx.cache.access_levels.is_public(did) {
                return;
            }

            if let Some(stab) = tcx.lookup_stability(did) {
                if stab.is_unstable() && stab.feature == sym::rustc_private {
                    return;
                }
            }
        }
    }

    let document_hidden = cx.render_options.document_hidden;
    let predicates = tcx.explicit_predicates_of(did);
    let (trait_items, generics) = match impl_item {
        Some(impl_) => (
            impl_
                .items
                .iter()
                .map(|item| tcx.hir().impl_item(item.id))
                .filter(|item| {
                    // Filter out impl items whose corresponding trait item has `doc(hidden)`
                    // not to document such impl items.
                    // For inherent impls, we don't do any filtering, because that's already done in strip_hidden.rs.

                    // When `--document-hidden-items` is passed, we don't
                    // do any filtering, too.
                    if document_hidden {
                        return true;
                    }
                    if let Some(associated_trait) = associated_trait {
                        let assoc_kind = match item.kind {
                            hir::ImplItemKind::Const(..) => ty::AssocKind::Const,
                            hir::ImplItemKind::Fn(..) => ty::AssocKind::Fn,
                            hir::ImplItemKind::TyAlias(..) => ty::AssocKind::Type,
                        };
                        let trait_item = tcx
                            .associated_items(associated_trait.def_id)
                            .find_by_name_and_kind(
                                tcx,
                                item.ident,
                                assoc_kind,
                                associated_trait.def_id,
                            )
                            .unwrap(); // SAFETY: For all impl items there exists trait item that has the same name.
                        !tcx.is_doc_hidden(trait_item.def_id)
                    } else {
                        true
                    }
                })
                .map(|item| clean_impl_item(item, cx))
                .collect::<Vec<_>>(),
            clean_generics(impl_.generics, cx),
        ),
        None => (
            tcx.associated_items(did)
                .in_definition_order()
                .filter(|item| {
                    // If this is a trait impl, filter out associated items whose corresponding item
                    // in the associated trait is marked `doc(hidden)`.
                    // If this is an inherent impl, filter out private associated items.
                    if let Some(associated_trait) = associated_trait {
                        let trait_item = tcx
                            .associated_items(associated_trait.def_id)
                            .find_by_name_and_kind(
                                tcx,
                                item.ident(tcx),
                                item.kind,
                                associated_trait.def_id,
                            )
                            .unwrap(); // corresponding associated item has to exist
                        !tcx.is_doc_hidden(trait_item.def_id)
                    } else {
                        item.visibility(tcx).is_public()
                    }
                })
                .map(|item| clean_middle_assoc_item(item, cx))
                .collect::<Vec<_>>(),
            clean::enter_impl_trait(cx, |cx| {
                clean_ty_generics(cx, tcx.generics_of(did), predicates)
            }),
        ),
    };
    let polarity = tcx.impl_polarity(did);
    let trait_ = associated_trait.map(|t| clean_trait_ref_with_bindings(cx, t, &[]));
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        super::build_deref_target_impls(cx, &trait_items, ret);
    }

    // Return if the trait itself or any types of the generic parameters are doc(hidden).
    let mut stack: Vec<&Type> = vec![&for_];

    if let Some(did) = trait_.as_ref().map(|t| t.def_id()) {
        if tcx.is_doc_hidden(did) {
            return;
        }
    }
    if let Some(generics) = trait_.as_ref().and_then(|t| t.generics()) {
        stack.extend(generics);
    }

    while let Some(ty) = stack.pop() {
        if let Some(did) = ty.def_id(&cx.cache) {
            if tcx.is_doc_hidden(did) {
                return;
            }
        }
        if let Some(generics) = ty.generics() {
            stack.extend(generics);
        }
    }

    if let Some(did) = trait_.as_ref().map(|t| t.def_id()) {
        record_extern_trait(cx, did);
    }

    let (merged_attrs, cfg) = merge_attrs(cx, parent_module, load_attrs(cx, did), attrs);
    trace!("merged_attrs={:?}", merged_attrs);

    trace!(
        "build_impl: impl {:?} for {:?}",
        trait_.as_ref().map(|t| t.def_id()),
        for_.def_id(&cx.cache)
    );
    ret.push(clean::Item::from_def_id_and_attrs_and_parts(
        did,
        None,
        clean::ImplItem(Box::new(clean::Impl {
            unsafety: hir::Unsafety::Normal,
            generics,
            trait_,
            for_,
            items: trait_items,
            polarity,
            kind: if utils::has_doc_flag(tcx, did, sym::fake_variadic) {
                ImplKind::FakeVaradic
            } else {
                ImplKind::Normal
            },
        })),
        Box::new(merged_attrs),
        cx,
        cfg,
    ));
}

fn build_module(
    cx: &mut DocContext<'_>,
    did: DefId,
    visited: &mut FxHashSet<DefId>,
) -> clean::Module {
    let items = build_module_items(cx, did, visited, &mut FxHashSet::default());

    let span = clean::Span::new(cx.tcx.def_span(did));
    clean::Module { items, span }
}

fn build_module_items(
    cx: &mut DocContext<'_>,
    did: DefId,
    visited: &mut FxHashSet<DefId>,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Vec<clean::Item> {
    let mut items = Vec::new();

    // If we're re-exporting a re-export it may actually re-export something in
    // two namespaces, so the target may be listed twice. Make sure we only
    // visit each node at most once.
    for &item in cx.tcx.module_children(did).iter() {
        if item.vis.is_public() {
            let res = item.res.expect_non_local();
            if let Some(def_id) = res.mod_def_id() {
                // If we're inlining a glob import, it's possible to have
                // two distinct modules with the same name. We don't want to
                // inline it, or mark any of its contents as visited.
                if did == def_id
                    || inlined_names.contains(&(ItemType::Module, item.ident.name))
                    || !visited.insert(def_id)
                {
                    continue;
                }
            }
            if let Res::PrimTy(p) = res {
                // Primitive types can't be inlined so generate an import instead.
                let prim_ty = clean::PrimitiveType::from(p);
                items.push(clean::Item {
                    name: None,
                    attrs: Box::new(clean::Attributes::default()),
                    item_id: ItemId::Primitive(prim_ty, did.krate),
                    visibility: clean::Public,
                    kind: Box::new(clean::ImportItem(clean::Import::new_simple(
                        item.ident.name,
                        clean::ImportSource {
                            path: clean::Path {
                                res,
                                segments: vec![clean::PathSegment {
                                    name: prim_ty.as_sym(),
                                    args: clean::GenericArgs::AngleBracketed {
                                        args: Default::default(),
                                        bindings: ThinVec::new(),
                                    },
                                }],
                            },
                            did: None,
                        },
                        true,
                    ))),
                    cfg: None,
                });
            } else if let Some(i) = try_inline(cx, did, None, res, item.ident.name, None, visited) {
                items.extend(i)
            }
        }
    }

    items
}

pub(crate) fn print_inlined_const(tcx: TyCtxt<'_>, did: DefId) -> String {
    if let Some(did) = did.as_local() {
        let hir_id = tcx.hir().local_def_id_to_hir_id(did);
        rustc_hir_pretty::id_to_string(&tcx.hir(), hir_id)
    } else {
        tcx.rendered_const(did).clone()
    }
}

fn build_const(cx: &mut DocContext<'_>, def_id: DefId) -> clean::Constant {
    clean::Constant {
        type_: clean_middle_ty(cx.tcx.type_of(def_id), cx, Some(def_id)),
        kind: clean::ConstantKind::Extern { def_id },
    }
}

fn build_static(cx: &mut DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: clean_middle_ty(cx.tcx.type_of(did), cx, Some(did)),
        mutability: if mutable { Mutability::Mut } else { Mutability::Not },
        expr: None,
    }
}

fn build_macro(
    cx: &mut DocContext<'_>,
    def_id: DefId,
    name: Symbol,
    import_def_id: Option<DefId>,
) -> clean::ItemKind {
    match CStore::from_tcx(cx.tcx).load_macro_untracked(def_id, cx.sess()) {
        LoadedMacro::MacroDef(item_def, _) => {
            if let ast::ItemKind::MacroDef(ref def) = item_def.kind {
                let vis = clean_visibility(cx.tcx.visibility(import_def_id.unwrap_or(def_id)));
                clean::MacroItem(clean::Macro {
                    source: utils::display_macro_source(cx, name, def, def_id, vis),
                })
            } else {
                unreachable!()
            }
        }
        LoadedMacro::ProcMacro(ext) => clean::ProcMacroItem(clean::ProcMacro {
            kind: ext.macro_kind(),
            helpers: ext.helper_attrs,
        }),
    }
}

/// A trait's generics clause actually contains all of the predicates for all of
/// its associated types as well. We specifically move these clauses to the
/// associated types instead when displaying, so when we're generating the
/// generics for the trait itself we need to be sure to remove them.
/// We also need to remove the implied "recursive" Self: Trait bound.
///
/// The inverse of this filtering logic can be found in the `Clean`
/// implementation for `AssociatedType`
fn filter_non_trait_generics(trait_did: DefId, mut g: clean::Generics) -> clean::Generics {
    for pred in &mut g.where_predicates {
        match *pred {
            clean::WherePredicate::BoundPredicate {
                ty: clean::Generic(ref s),
                ref mut bounds,
                ..
            } if *s == kw::SelfUpper => {
                bounds.retain(|bound| match bound {
                    clean::GenericBound::TraitBound(clean::PolyTrait { trait_, .. }, _) => {
                        trait_.def_id() != trait_did
                    }
                    _ => true,
                });
            }
            _ => {}
        }
    }

    g.where_predicates.retain(|pred| match pred {
        clean::WherePredicate::BoundPredicate {
            ty: clean::QPath(box clean::QPathData { self_type: clean::Generic(ref s), trait_, .. }),
            bounds,
            ..
        } => !(bounds.is_empty() || *s == kw::SelfUpper && trait_.def_id() == trait_did),
        _ => true,
    });
    g
}

/// Supertrait bounds for a trait are also listed in the generics coming from
/// the metadata for a crate, so we want to separate those out and create a new
/// list of explicit supertrait bounds to render nicely.
fn separate_supertrait_bounds(
    mut g: clean::Generics,
) -> (clean::Generics, Vec<clean::GenericBound>) {
    let mut ty_bounds = Vec::new();
    g.where_predicates.retain(|pred| match *pred {
        clean::WherePredicate::BoundPredicate { ty: clean::Generic(ref s), ref bounds, .. }
            if *s == kw::SelfUpper =>
        {
            ty_bounds.extend(bounds.iter().cloned());
            false
        }
        _ => true,
    });
    (g, ty_bounds)
}

pub(crate) fn record_extern_trait(cx: &mut DocContext<'_>, did: DefId) {
    if did.is_local() {
        return;
    }

    {
        if cx.external_traits.borrow().contains_key(&did) || cx.active_extern_traits.contains(&did)
        {
            return;
        }
    }

    {
        cx.active_extern_traits.insert(did);
    }

    debug!("record_extern_trait: {:?}", did);
    let trait_ = build_external_trait(cx, did);

    let trait_ = clean::TraitWithExtraInfo {
        trait_,
        is_notable: clean::utils::has_doc_flag(cx.tcx, did, sym::notable_trait),
    };
    cx.external_traits.borrow_mut().insert(did, trait_);
    cx.active_extern_traits.remove(&did);
}
