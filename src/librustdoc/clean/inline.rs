//! Support for inlining external documentation into the current AST.

use std::iter::once;
use std::sync::Arc;

use thin_vec::{thin_vec, ThinVec};

use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdSet, LocalDefId};
use rustc_hir::Mutability;
use rustc_metadata::creader::{CStore, LoadedMacro};
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{kw, sym, Symbol};

use crate::clean::{
    self, clean_fn_decl_from_did_and_sig, clean_generics, clean_impl_item, clean_middle_assoc_item,
    clean_middle_field, clean_middle_ty, clean_trait_ref_with_bindings, clean_ty,
    clean_ty_generics, clean_variant_def, utils, Attributes, AttributesExt, ImplKind, ItemId, Type,
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
pub(crate) fn try_inline(
    cx: &mut DocContext<'_>,
    res: Res,
    name: Symbol,
    attrs: Option<(&[ast::Attribute], Option<DefId>)>,
    visited: &mut DefIdSet,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }
    let mut ret = Vec::new();

    debug!("attrs={:?}", attrs);

    let attrs_without_docs = attrs.map(|(attrs, def_id)| {
        (attrs.into_iter().filter(|a| a.doc_str().is_none()).cloned().collect::<Vec<_>>(), def_id)
    });
    let attrs_without_docs =
        attrs_without_docs.as_ref().map(|(attrs, def_id)| (&attrs[..], *def_id));

    let import_def_id = attrs.and_then(|(_, def_id)| def_id);
    let kind = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, ItemType::Trait);
            build_impls(cx, did, attrs_without_docs, &mut ret);
            clean::TraitItem(Box::new(build_external_trait(cx, did)))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, ItemType::Function);
            clean::FunctionItem(build_external_function(cx, did))
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, ItemType::Struct);
            build_impls(cx, did, attrs_without_docs, &mut ret);
            clean::StructItem(build_struct(cx, did))
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, ItemType::Union);
            build_impls(cx, did, attrs_without_docs, &mut ret);
            clean::UnionItem(build_union(cx, did))
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, ItemType::Typedef);
            build_impls(cx, did, attrs_without_docs, &mut ret);
            clean::TypedefItem(build_type_alias(cx, did))
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, ItemType::Enum);
            build_impls(cx, did, attrs_without_docs, &mut ret);
            clean::EnumItem(build_enum(cx, did))
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, ItemType::ForeignType);
            build_impls(cx, did, attrs_without_docs, &mut ret);
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
            let mac = build_macro(cx, did, name, import_def_id, kind);

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

    let (attrs, cfg) = merge_attrs(cx, load_attrs(cx, did), attrs);
    cx.inlined.insert(did.into());
    let mut item =
        clean::Item::from_def_id_and_attrs_and_parts(did, Some(name), kind, Box::new(attrs), cfg);
    // The visibility needs to reflect the one from the reexport and not from the "source" DefId.
    item.inline_stmt_id = import_def_id;
    ret.push(item);
    Some(ret)
}

pub(crate) fn try_inline_glob(
    cx: &mut DocContext<'_>,
    res: Res,
    current_mod: LocalDefId,
    visited: &mut DefIdSet,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }

    match res {
        Res::Def(DefKind::Mod, did) => {
            // Use the set of module reexports to filter away names that are not actually
            // reexported by the glob, e.g. because they are shadowed by something else.
            let reexports = cx
                .tcx
                .module_children_local(current_mod)
                .iter()
                .filter(|child| !child.reexport_chain.is_empty())
                .filter_map(|child| child.res.opt_def_id())
                .collect();
            let mut items = build_module_items(cx, did, visited, inlined_names, Some(&reexports));
            items.retain(|item| {
                if let Some(name) = item.name {
                    // If an item with the same type and name already exists,
                    // it takes priority over the inlined stuff.
                    inlined_names.insert((item.type_(), name))
                } else {
                    true
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
        .map(|item| clean_middle_assoc_item(item, cx))
        .collect();

    let predicates = cx.tcx.predicates_of(did);
    let generics = clean_ty_generics(cx, cx.tcx.generics_of(did), predicates);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    clean::Trait { def_id: did, generics, items: trait_items, bounds: supertrait_bounds }
}

fn build_external_function<'tcx>(cx: &mut DocContext<'tcx>, did: DefId) -> Box<clean::Function> {
    let sig = cx.tcx.fn_sig(did).instantiate_identity();

    let late_bound_regions = sig.bound_vars().into_iter().filter_map(|var| match var {
        ty::BoundVariableKind::Region(ty::BrNamed(_, name)) if name != kw::UnderscoreLifetime => {
            Some(clean::GenericParamDef::lifetime(name))
        }
        _ => None,
    });

    let predicates = cx.tcx.explicit_predicates_of(did);
    let (generics, decl) = clean::enter_impl_trait(cx, |cx| {
        // NOTE: generics need to be cleaned before the decl!
        let mut generics = clean_ty_generics(cx, cx.tcx.generics_of(did), predicates);
        // FIXME: This does not place parameters in source order (late-bound ones come last)
        generics.params.extend(late_bound_regions);
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
        ctor_kind: variant.ctor_kind(),
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
    let type_ = clean_middle_ty(
        ty::Binder::dummy(cx.tcx.type_of(did).instantiate_identity()),
        cx,
        Some(did),
        None,
    );

    Box::new(clean::Typedef {
        type_,
        generics: clean_ty_generics(cx, cx.tcx.generics_of(did), predicates),
        item_type: None,
    })
}

/// Builds all inherent implementations of an ADT (struct/union/enum) or Trait item/path/reexport.
pub(crate) fn build_impls(
    cx: &mut DocContext<'_>,
    did: DefId,
    attrs: Option<(&[ast::Attribute], Option<DefId>)>,
    ret: &mut Vec<clean::Item>,
) {
    let _prof_timer = cx.tcx.sess.prof.generic_activity("build_inherent_impls");
    let tcx = cx.tcx;

    // for each implementation of an item represented by `did`, build the clean::Item for that impl
    for &did in tcx.inherent_impls(did).iter() {
        build_impl(cx, did, attrs, ret);
    }

    // This pretty much exists expressly for `dyn Error` traits that exist in the `alloc` crate.
    // See also:
    //
    // * https://github.com/rust-lang/rust/issues/103170 — where it didn't used to get documented
    // * https://github.com/rust-lang/rust/pull/99917 — where the feature got used
    // * https://github.com/rust-lang/rust/issues/53487 — overall tracking issue for Error
    if tcx.has_attr(did, sym::rustc_has_incoherent_inherent_impls) {
        let type_ =
            if tcx.is_trait(did) { SimplifiedType::Trait(did) } else { SimplifiedType::Adt(did) };
        for &did in tcx.incoherent_impls(type_) {
            build_impl(cx, did, attrs, ret);
        }
    }
}

pub(crate) fn merge_attrs(
    cx: &mut DocContext<'_>,
    old_attrs: &[ast::Attribute],
    new_attrs: Option<(&[ast::Attribute], Option<DefId>)>,
) -> (clean::Attributes, Option<Arc<clean::cfg::Cfg>>) {
    // NOTE: If we have additional attributes (from a re-export),
    // always insert them first. This ensure that re-export
    // doc comments show up before the original doc comments
    // when we render them.
    if let Some((inner, item_id)) = new_attrs {
        let mut both = inner.to_vec();
        both.extend_from_slice(old_attrs);
        (
            if let Some(item_id) = item_id {
                Attributes::from_ast_with_additional(old_attrs, (inner, item_id))
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
    did: DefId,
    attrs: Option<(&[ast::Attribute], Option<DefId>)>,
    ret: &mut Vec<clean::Item>,
) {
    if !cx.inlined.insert(did.into()) {
        return;
    }

    let tcx = cx.tcx;
    let _prof_timer = tcx.sess.prof.generic_activity("build_impl");

    let associated_trait = tcx.impl_trait_ref(did).map(ty::EarlyBinder::skip_binder);

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local() && let Some(traitref) = associated_trait {
        let did = traitref.def_id;
        if !cx.cache.effective_visibilities.is_directly_public(tcx, did) {
            return;
        }

        if let Some(stab) = tcx.lookup_stability(did) &&
            stab.is_unstable() &&
            stab.feature == sym::rustc_private
        {
            return;
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
        None => clean_middle_ty(
            ty::Binder::dummy(tcx.type_of(did).instantiate_identity()),
            cx,
            Some(did),
            None,
        ),
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(did) = for_.def_id(&cx.cache) {
            if !cx.cache.effective_visibilities.is_directly_public(tcx, did) {
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
                            hir::ImplItemKind::Type(..) => ty::AssocKind::Type,
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
                        document_hidden || !tcx.is_doc_hidden(trait_item.def_id)
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
    let trait_ = associated_trait
        .map(|t| clean_trait_ref_with_bindings(cx, ty::Binder::dummy(t), ThinVec::new()));
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        super::build_deref_target_impls(cx, &trait_items, ret);
    }

    // Return if the trait itself or any types of the generic parameters are doc(hidden).
    let mut stack: Vec<&Type> = vec![&for_];

    if let Some(did) = trait_.as_ref().map(|t| t.def_id()) {
        if !document_hidden && tcx.is_doc_hidden(did) {
            return;
        }
    }
    if let Some(generics) = trait_.as_ref().and_then(|t| t.generics()) {
        stack.extend(generics);
    }

    while let Some(ty) = stack.pop() {
        if let Some(did) = ty.def_id(&cx.cache) && !document_hidden && tcx.is_doc_hidden(did) {
            return;
        }
        if let Some(generics) = ty.generics() {
            stack.extend(generics);
        }
    }

    if let Some(did) = trait_.as_ref().map(|t| t.def_id()) {
        record_extern_trait(cx, did);
    }

    let (merged_attrs, cfg) = merge_attrs(cx, load_attrs(cx, did), attrs);
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
                ImplKind::FakeVariadic
            } else {
                ImplKind::Normal
            },
        })),
        Box::new(merged_attrs),
        cfg,
    ));
}

fn build_module(cx: &mut DocContext<'_>, did: DefId, visited: &mut DefIdSet) -> clean::Module {
    let items = build_module_items(cx, did, visited, &mut FxHashSet::default(), None);

    let span = clean::Span::new(cx.tcx.def_span(did));
    clean::Module { items, span }
}

fn build_module_items(
    cx: &mut DocContext<'_>,
    did: DefId,
    visited: &mut DefIdSet,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
    allowed_def_ids: Option<&DefIdSet>,
) -> Vec<clean::Item> {
    let mut items = Vec::new();

    // If we're re-exporting a re-export it may actually re-export something in
    // two namespaces, so the target may be listed twice. Make sure we only
    // visit each node at most once.
    for item in cx.tcx.module_children(did).iter() {
        if item.vis.is_public() {
            let res = item.res.expect_non_local();
            if let Some(def_id) = res.opt_def_id()
                && let Some(allowed_def_ids) = allowed_def_ids
                && !allowed_def_ids.contains(&def_id) {
                continue;
            }
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
                    // We can use the item's `DefId` directly since the only information ever used
                    // from it is `DefId.krate`.
                    item_id: ItemId::DefId(did),
                    kind: Box::new(clean::ImportItem(clean::Import::new_simple(
                        item.ident.name,
                        clean::ImportSource {
                            path: clean::Path {
                                res,
                                segments: thin_vec![clean::PathSegment {
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
                    inline_stmt_id: None,
                });
            } else if let Some(i) = try_inline(cx, res, item.ident.name, None, visited) {
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
        type_: clean_middle_ty(
            ty::Binder::dummy(cx.tcx.type_of(def_id).instantiate_identity()),
            cx,
            Some(def_id),
            None,
        ),
        kind: clean::ConstantKind::Extern { def_id },
    }
}

fn build_static(cx: &mut DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: clean_middle_ty(
            ty::Binder::dummy(cx.tcx.type_of(did).instantiate_identity()),
            cx,
            Some(did),
            None,
        ),
        mutability: if mutable { Mutability::Mut } else { Mutability::Not },
        expr: None,
    }
}

fn build_macro(
    cx: &mut DocContext<'_>,
    def_id: DefId,
    name: Symbol,
    import_def_id: Option<DefId>,
    macro_kind: MacroKind,
) -> clean::ItemKind {
    match CStore::from_tcx(cx.tcx).load_macro_untracked(def_id, cx.sess()) {
        LoadedMacro::MacroDef(item_def, _) => match macro_kind {
            MacroKind::Bang => {
                if let ast::ItemKind::MacroDef(ref def) = item_def.kind {
                    let vis = cx.tcx.visibility(import_def_id.unwrap_or(def_id));
                    clean::MacroItem(clean::Macro {
                        source: utils::display_macro_source(cx, name, def, def_id, vis),
                    })
                } else {
                    unreachable!()
                }
            }
            MacroKind::Derive | MacroKind::Attr => {
                clean::ProcMacroItem(clean::ProcMacro { kind: macro_kind, helpers: Vec::new() })
            }
        },
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
            ty:
                clean::QPath(box clean::QPathData {
                    self_type: clean::Generic(ref s),
                    trait_: Some(trait_),
                    ..
                }),
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

    cx.external_traits.borrow_mut().insert(did, trait_);
    cx.active_extern_traits.remove(&did);
}
